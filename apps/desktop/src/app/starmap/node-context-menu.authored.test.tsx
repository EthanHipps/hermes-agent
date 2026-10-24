import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, expect, it, vi } from 'vitest'

import { editLearningNode } from '@/api/skills'
import type { HermesApiRequest } from '@/global'
import { NodeContextMenu } from './node-context-menu'

vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => null }))
vi.mock('@/store/starmap', () => ({ loadStarmapGraph: vi.fn(), evictStarmapNode: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('../hooks/use-on-profile-switch', () => ({ useOnProfileSwitch: () => undefined }))
vi.mock('@/app/learning/archive-skill-confirm-dialog', () => ({ ArchiveSkillConfirmDialog: () => null, fireOptimistic: vi.fn() }))
vi.mock('@/components/ui/confirm-dialog', () => ({ ConfirmDialog: () => null }))
vi.mock('@/components/ui/button', () => ({ Button: ({children, ...props}: {children: ReactNode}) => <button {...props}>{children}</button> }))
vi.mock('@/components/ui/dialog', () => ({
  Dialog: ({open, children}: {open: boolean; children: ReactNode}) => open ? children : null,
  DialogContent: ({children}: {children: ReactNode}) => <div>{children}</div>,
  DialogFooter: ({children}: {children: ReactNode}) => <div>{children}</div>,
  DialogHeader: ({children}: {children: ReactNode}) => <div>{children}</div>,
  DialogTitle: ({children}: {children: ReactNode}) => <div>{children}</div>
}))
vi.mock('@/hermes', async () => await import('@/api/skills'))
vi.mock('@/i18n/context', () => ({ useI18n: () => ({ t: { memoryAdmission: {
  label: 'Use this memory as', declarative: 'Facts and observations', promptLike: 'Instructions or prompts'
} } }) }))
let root: Root | undefined
let host: HTMLDivElement | undefined
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
afterEach(async () => { await act(async () => root?.unmount()); host?.remove(); root = undefined; delete (window as {hermesDesktop?: unknown}).hermesDesktop })

it.each(['declarative', 'prompt_like'] as const)('preserves %s in the actual request builder', async choice => {
  const requests: HermesApiRequest[] = []
  ;(window as {hermesDesktop?: unknown}).hermesDesktop = { api: async (request: HermesApiRequest) => { requests.push(request); return {ok: true} } }
  await editLearningNode('memory:memory:r1', 'authored', {connectionId: 'remote', profile: 'writer'}, choice)
  expect(requests).toEqual([{path: '/api/learning/node', method: 'PUT', connectionId: 'remote', profile: 'writer',
    body: {id: 'memory:memory:r1', content: 'authored', memory_admission: choice}}])
})

it.each(['memory', 'profile'] as const)('authors an explicit choice only for %s', async source => {
  const requests: HermesApiRequest[] = []
  ;(window as {hermesDesktop?: unknown}).hermesDesktop = { api: async (request: HermesApiRequest) => {
    requests.push(request)
    return request.method === 'PUT' ? {ok: true} : {ok: true, kind: 'memory', label: 'fact', content: 'authored'}
  } }
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
  await act(async () => root!.render(<NodeContextMenu onClose={() => undefined} onNodeRemoved={() => undefined}
    target={{id: `memory:${source}:r1`, kind: 'memory', label: 'fact', x: 0, y: 0}} />))
  const button = (name: string) => Array.from(host!.querySelectorAll('button')).find(b => b.textContent === name)!
  await act(async () => button('Edit memory…').click())
  if (source === 'memory') {
    expect(button('Save').disabled).toBe(true)
    expect(button('Facts and observations').getAttribute('aria-pressed')).toBe('false')
    expect(button('Instructions or prompts').getAttribute('aria-pressed')).toBe('false')
    await act(async () => button('Instructions or prompts').click())
  }
  await act(async () => button('Save').click())
  expect(requests.find(request => request.method === 'PUT')?.body).toEqual({id: `memory:${source}:r1`, content: 'authored',
    memory_admission: source === 'memory' ? 'prompt_like' : null})
  if (source === 'memory') {
    await act(async () => button('Edit memory…').click())
    expect(button('Save').disabled).toBe(true)
    expect(button('Instructions or prompts').getAttribute('aria-pressed')).toBe('false')
    await act(async () => button('Facts and observations').click())
    await act(async () => button('Cancel').click())
    await act(async () => button('Edit memory…').click())
    expect(button('Save').disabled).toBe(true)
    expect(button('Facts and observations').getAttribute('aria-pressed')).toBe('false')
  }
})
