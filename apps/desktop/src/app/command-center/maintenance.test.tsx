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

function httpError(statusCode: number, detail: unknown) {
  return Object.assign(new Error(`${statusCode}: ${JSON.stringify({ detail })}`), { statusCode })
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

  it('never offers a reset in authoritative mode without a scope the status gave', async () => {
    mocks.getMemoryStatus.mockResolvedValue({ ...AUTHORITATIVE, builtin_files: { memory: 4096, user: 4096 } })
    render(<MaintenancePanel />)

    const memoryReset = await screen.findByRole('button', { name: 'Reset memory' })
    const userReset = screen.getByRole('button', { name: 'Reset profile' })

    expect(memoryReset).toHaveProperty('disabled', true)
    expect(userReset).toHaveProperty('disabled', true)
    expect(screen.getByText(/provider-managed/i)).toBeTruthy()
    fireEvent.click(memoryReset)
    expect(mocks.confirm).not.toHaveBeenCalled()
    expect(mocks.resetMemory).not.toHaveBeenCalled()
  })

  it('does not call the native files "deleted" in authoritative mode', async () => {
    mocks.getMemoryStatus.mockResolvedValue(AUTHORITATIVE)
    render(<MaintenancePanel />)

    await screen.findByRole('button', { name: 'Reset memory' })

    expect(screen.queryByText(/MEMORY\.md/)).toBeNull()
    expect(screen.queryByText(/USER\.md/)).toBeNull()
  })
})

describe('runMemoryReset', () => {
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
