import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { MemoryStatusResponse } from '@/types/hermes'

import { MaintenancePanel } from './maintenance'
import { runMemoryReset } from './memory-reset'

const mocks = vi.hoisted(() => ({
  confirm: vi.fn(() => Promise.resolve(true)),
  getMemoryStatus: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn(),
  resetMemory: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: vi.fn(() => Promise.resolve({ running: false })),
  getCuratorStatus: vi.fn(() => Promise.reject(new Error('no curator'))),
  getMemoryStatus: mocks.getMemoryStatus,
  resetMemory: mocks.resetMemory
}))
vi.mock('@/store/confirm', () => ({ confirm: mocks.confirm }))
vi.mock('@/store/notifications', () => ({ notify: mocks.notify, notifyError: mocks.notifyError }))
vi.mock('@/store/activity', () => ({ upsertDesktopActionTask: vi.fn() }))

const ADDITIVE: MemoryStatusResponse = {
  active: '',
  providers: [],
  builtin_files: { memory: 0, user: 2048 }
}

const AUTHORITATIVE: MemoryStatusResponse = {
  active: 'ygg',
  providers: [],
  builtin_files: { memory: 0, user: 0 },
  curated_memory: {
    authority: 'provider',
    provider: 'ygg',
    provider_api: 1,
    included: false,
    disposition: 'provider-managed',
    restore_action: 'reconnect-provider',
    provider_mode: 'authoritative',
    failure_policy: 'fail_closed',
    targets: { memory: true, user: true },
    configuration_error: null
  }
}

// What the renderer really receives: ipcRenderer.invoke keeps only the message, wrapped, with no statusCode.
function httpError(statusCode: number, detail: unknown) {
  return new Error(
    `Error invoking remote method 'hermes:api': Error: ${statusCode}: ${JSON.stringify({ detail })}`
  )
}

beforeEach(() => {
  mocks.confirm.mockClear()
  mocks.notify.mockClear()
  mocks.notifyError.mockClear()
  mocks.resetMemory.mockReset()
  mocks.getMemoryStatus.mockReset()
})

afterEach(cleanup)

describe('Maintenance memory reset', () => {
  it('keeps the additive reset body and deleted toast exactly as before', async () => {
    mocks.getMemoryStatus.mockResolvedValue(ADDITIVE)
    mocks.resetMemory.mockResolvedValue({ ok: true, deleted: ['USER.md'] })
    render(<MaintenancePanel />)

    fireEvent.click(await screen.findByRole('button', { name: 'Reset profile' }))

    await waitFor(() => expect(mocks.resetMemory).toHaveBeenCalledTimes(1))
    expect(mocks.resetMemory).toHaveBeenCalledWith({ target: 'user' })
    await waitFor(() =>
      expect(mocks.notify).toHaveBeenCalledWith({ kind: 'success', title: 'Deleted USER.md.', message: '' })
    )
  })

  async function openForm(button: string) {
    mocks.getMemoryStatus.mockResolvedValue(AUTHORITATIVE)
    render(<MaintenancePanel />)
    fireEvent.click(await screen.findByRole('button', { name: button }))
  }

  const type = (label: string, value: string) =>
    fireEvent.change(screen.getByLabelText(label), { target: { value } })

  it('opens a scope form, not the additive confirm, and cannot submit without a scope id', async () => {
    await openForm('Reset memory')

    const submit = screen.getByRole('button', { name: 'Request reset' })

    expect(mocks.confirm).not.toHaveBeenCalled()
    expect(submit).toHaveProperty('disabled', true)
    fireEvent.click(submit)
    expect(mocks.resetMemory).not.toHaveBeenCalled()
  })

  it('sends the exact scoped body once the form is filled', async () => {
    mocks.resetMemory.mockResolvedValue({ ok: false, code: 'approval_pending', pending_id: 'abc123def456' })
    await openForm('Reset memory')
    type('Scope kind', 'repository')
    type('Scope id', 'repo-1')
    type('Project id (project_id)', 'proj-1')
    type('Repository id (repo_id)', 'repo-1')
    fireEvent.click(screen.getByRole('button', { name: 'Request reset' }))

    await waitFor(() => expect(mocks.resetMemory).toHaveBeenCalledTimes(1))
    expect(mocks.resetMemory).toHaveBeenCalledWith({
      target: 'memory',
      scopes: ['repository:repo-1'],
      project_id: 'proj-1',
      repo_id: 'repo-1'
    })
    await waitFor(() =>
      expect(mocks.notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'info', title: expect.stringContaining('abc123def456') }))
    )
  })

  it('offers only the four scope kinds and never the all target', async () => {
    await openForm('Reset profile')

    const kinds = Array.from(screen.getByLabelText('Scope kind').querySelectorAll('option')).map(o => o.value)

    expect(kinds).toEqual(['global', 'organization', 'project', 'repository'])
    expect(screen.queryByText(/reset both/i)).toBeNull()
    type('Scope kind', 'global')
    type('Scope id', 'ethan')
    fireEvent.click(screen.getByRole('button', { name: 'Request reset' }))
    await waitFor(() => expect(mocks.resetMemory).toHaveBeenCalledTimes(1))
    expect(mocks.resetMemory).toHaveBeenCalledWith({ target: 'user', scopes: ['global:ethan'] })
  })

  it('shows the backend 400 as a refused outcome, not a crash or a deletion', async () => {
    mocks.resetMemory.mockRejectedValue(httpError(400, { error: 'unauthorized_scope', message: 'scope not allowed' }))
    await openForm('Reset memory')
    type('Scope id', 'repo-1')
    fireEvent.click(screen.getByRole('button', { name: 'Request reset' }))

    await waitFor(() =>
      expect(mocks.notify).toHaveBeenCalledWith({ kind: 'error', title: 'Memory reset failed', message: 'scope not allowed' })
    )
    expect(mocks.notify).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
  })

  it('shows a 409 as a refusal that changed nothing', async () => {
    mocks.resetMemory.mockRejectedValue(httpError(409, { error: 'approval_unavailable', message: 'nothing was changed' }))
    await openForm('Reset memory')
    type('Scope id', 'repo-1')
    fireEvent.click(screen.getByRole('button', { name: 'Request reset' }))

    await waitFor(() =>
      expect(mocks.notify).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'error', title: 'Memory reset failed', message: expect.stringContaining('nothing was changed') })
      )
    )
    expect(mocks.notify).not.toHaveBeenCalledWith(expect.objectContaining({ message: expect.stringContaining('Error invoking') }))
    expect(mocks.notify).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
  })

  it('shows a 503 as an unknown outcome, never as a plain failure or a deletion', async () => {
    mocks.resetMemory.mockRejectedValue(httpError(503, { error: 'memory_unavailable', message: 'provider down' }))
    await openForm('Reset memory')
    type('Scope id', 'repo-1')
    fireEvent.click(screen.getByRole('button', { name: 'Request reset' }))

    await waitFor(() =>
      expect(mocks.notify).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'warning', title: 'Memory reset failed', message: expect.stringContaining('provider down') })
      )
    )
    expect(mocks.notify).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
  })

  it('does not call the native files "deleted" in authoritative mode', async () => {
    mocks.getMemoryStatus.mockResolvedValue(AUTHORITATIVE)
    render(<MaintenancePanel />)

    await screen.findByRole('button', { name: 'Reset memory' })

    expect(screen.queryByText(/injected into every session/)).toBeNull()
    expect(screen.queryByText(/MEMORY\.md/)).toBeNull()
    expect(screen.queryByText(/USER\.md/)).toBeNull()
  })

  it('keeps the additive failure toast on notifyError', async () => {
    const failure = new Error("Error invoking remote method 'hermes:api': Error: 500: boom")

    mocks.getMemoryStatus.mockResolvedValue(ADDITIVE)
    mocks.resetMemory.mockRejectedValue(failure)
    render(<MaintenancePanel />)
    fireEvent.click(await screen.findByRole('button', { name: 'Reset profile' }))

    await waitFor(() => expect(mocks.notifyError).toHaveBeenCalledWith(failure, 'Memory reset failed'))
  })

  it('shows the additive description above the native files', async () => {
    mocks.getMemoryStatus.mockResolvedValue(ADDITIVE)
    render(<MaintenancePanel />)

    expect(await screen.findByText(/injected into every session/)).toBeTruthy()
  })
})

describe('runMemoryReset', () => {
  it.each([
    { error: httpError(500, 'boom'), expected: { kind: 'unavailable', message: 'boom' } },
    { error: httpError(499, 'nope'), expected: { kind: 'failed', message: 'nope' } },
    {
      error: new Error('connect ECONNREFUSED'),
      expected: { kind: 'unavailable', message: 'connect ECONNREFUSED' }
    }
  ])('classifies reset request errors', async ({ error, expected }) => {
    const outcome = await runMemoryReset(() => Promise.reject(error))

    expect(outcome).toEqual(expected)
  })

  it('reports a pending approval with its id', async () => {
    const outcome = await runMemoryReset(() =>
      Promise.resolve({ ok: false, code: 'approval_pending', pending_id: 'abc123def456' })
    )

    expect(outcome).toEqual({ kind: 'pending', pendingId: 'abc123def456' })
  })

  it('reports a typed refusal as refused', async () => {
    const outcome = await runMemoryReset(() =>
      Promise.reject(httpError(409, { error: 'approval_unavailable', message: 'nothing was changed' }))
    )

    expect(outcome).toEqual({ kind: 'refused', message: 'nothing was changed' })
  })

  it('reports an unavailable provider as an unknown outcome, not a deletion', async () => {
    const outcome = await runMemoryReset(() =>
      Promise.reject(httpError(503, { error: 'memory_unavailable', message: 'provider down' }))
    )

    expect(outcome).toEqual({ kind: 'unavailable', message: 'provider down' })
  })

  it('reports a rejected scope as failed with the backend message', async () => {
    const outcome = await runMemoryReset(() =>
      Promise.reject(httpError(400, { error: 'unauthorized_scope', message: 'scope not allowed' }))
    )

    expect(outcome).toEqual({ kind: 'failed', message: 'scope not allowed' })
  })

  it('does not claim deletion when the response has no deleted list', async () => {
    await expect(runMemoryReset(() => Promise.resolve({ ok: true } as never))).resolves.toEqual({ kind: 'unexpected' })
  })

  it('reports the deleted files when the backend lists them', async () => {
    await expect(runMemoryReset(() => Promise.resolve({ ok: true, deleted: ['MEMORY.md'] }))).resolves.toEqual({
      kind: 'deleted',
      deleted: ['MEMORY.md']
    })
  })
})
