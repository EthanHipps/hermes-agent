import type { MemoryResetResponse } from '@/types/hermes'

/** What a memory reset truly did, read from the backend's answer. */
export type MemoryResetOutcome =
  | { deleted: string[]; kind: 'deleted' }
  | { kind: 'failed'; message: string }
  | { kind: 'pending'; pendingId: string }
  | { kind: 'refused'; message: string }
  | { kind: 'unavailable'; message: string }
  | { kind: 'unexpected' }

/** A rejection reaches the renderer through `ipcRenderer.invoke`, which keeps only the message:
 *  `Error invoking remote method 'hermes:api': Error: 409: {"detail":...}`, with no `statusCode`.
 *  Unwrap that prefix and read the status from the leading `NNN:` of what remains. */
function errorBody(err: unknown): { message: string; status: number } {
  const raw = err instanceof Error ? err.message : String(err)
  const text = raw.match(/Error invoking remote method '[^']+': Error: (.+)$/s)?.[1] ?? raw
  const statusCode = (err as { statusCode?: unknown } | null)?.statusCode
  const prefixed = Number(text.match(/^(\d{3}):/)?.[1])
  const status = typeof statusCode === 'number' ? statusCode : prefixed
  const bodyText = text.replace(/^\d{3}:\s*/, '')

  try {
    const detail = (JSON.parse(bodyText) as { detail?: unknown }).detail

    if (typeof detail === 'string') {
      return { message: detail, status }
    }

    const message = (detail as { message?: unknown } | null)?.message

    if (typeof message === 'string') {
      return { message, status }
    }
  } catch {
    // Not JSON: fall through to the raw text.
  }

  return { message: text, status }
}

/** Runs a reset request and classifies its answer. Success is claimed only when the
 *  backend lists the files it deleted: a pending approval, a typed refusal, an unknown
 *  outcome and a body with no `deleted` each keep their own outcome. */
export async function runMemoryReset(request: () => Promise<MemoryResetResponse>): Promise<MemoryResetOutcome> {
  let result: MemoryResetResponse

  try {
    result = await request()
  } catch (err) {
    const { message, status } = errorBody(err)

    if (status === 409) {
      return { kind: 'refused', message }
    }

    return status === 503 ? { kind: 'unavailable', message } : { kind: 'failed', message }
  }

  if ('pending_id' in (result ?? {}) && (result as { code?: unknown }).code === 'approval_pending') {
    return { kind: 'pending', pendingId: (result as { pending_id: string }).pending_id }
  }

  const deleted = (result as { deleted?: unknown } | null)?.deleted

  return Array.isArray(deleted) ? { kind: 'deleted', deleted: deleted.map(String) } : { kind: 'unexpected' }
}
