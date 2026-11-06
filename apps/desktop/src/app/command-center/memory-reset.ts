import type { MemoryResetResponse } from '@/types/hermes'

/** What a memory reset truly did, read from the backend's answer. */
export type MemoryResetOutcome =
  | { deleted: string[]; kind: 'deleted' }
  | { kind: 'failed'; message: string }
  | { kind: 'pending'; pendingId: string }
  | { kind: 'refused'; message: string }
  | { kind: 'unavailable'; message: string }
  | { kind: 'unexpected' }

function errorBody(err: unknown): { message: string; status: number } {
  const text = err instanceof Error ? err.message : String(err)
  const statusCode = (err as { statusCode?: unknown } | null)?.statusCode
  const status = typeof statusCode === 'number' ? statusCode : NaN
  const bodyText = text.replace(/^\d+:\s*/, '')

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
