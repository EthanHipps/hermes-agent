import { afterEach, expect, it, vi } from 'vitest'

import { readKey, writeKey } from '@/lib/storage'

import {
  handoffReceiptKey,
  quarantineHandoffReceipt,
  readHandoffReceipt,
  saveStagedOnboardingFacts,
  stagedOnboardingFacts
} from './handoff-receipt'

afterEach(() => vi.restoreAllMocks())

it('preserves an unreadable receipt and removes it from the retry lookup', () => {
  const key = handoffReceiptKey('receipt-recovery', 'guide')
  const corrupt = '{unreadable receipt'
  writeKey(key, corrupt)

  expect(() => readHandoffReceipt(key)).toThrow()

  quarantineHandoffReceipt(key)

  expect(readKey(`${key}.unreadable`)).toBe(corrupt)
  expect(readHandoffReceipt(key)).toBeNull()
})

it.each(['throws', 'drops'] as const)('keeps the staged approval for retry when storage %s the write', failure => {
  const key = handoffReceiptKey(`pending-${failure}`, 'guide')
  const pendingId = `approval-${failure}`
  const storage = window.localStorage
  const storageOwner = Object.hasOwn(storage, 'setItem') ? storage : Object.getPrototypeOf(storage)

  const write = vi.spyOn(storageOwner, 'setItem').mockImplementation(() => {
    if (failure === 'throws') {
      throw new DOMException('Storage full', 'QuotaExceededError')
    }
  })

  expect(() => saveStagedOnboardingFacts(key, pendingId)).toThrow(pendingId)
  expect(readKey(`${key}.memory-pending`)).toBeNull()
  expect(stagedOnboardingFacts(key)).toBe(pendingId)
  expect(stagedOnboardingFacts(handoffReceiptKey('other-connection', 'guide'))).toBeNull()

  // A handoff retry must reuse and persist the existing request before creating its build session.
  write.mockRestore()
  saveStagedOnboardingFacts(key, stagedOnboardingFacts(key)!)
  expect(readKey(`${key}.memory-pending`)).toBe(pendingId)

  // Once persisted, the fallback no longer overrides the durable state.
  writeKey(`${key}.memory-pending`, null)
  expect(stagedOnboardingFacts(key)).toBeNull()
})
