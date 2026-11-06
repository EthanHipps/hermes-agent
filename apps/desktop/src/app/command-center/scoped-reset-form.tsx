import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import type { MemoryResetRequest } from '@/types/hermes'

/** The backend's four scope forms (`parse_scope_selector`). */
const SCOPE_KINDS = ['global', 'organization', 'project', 'repository'] as const

/** Provider-managed reset: one target, one explicit scope, and the administrative context.
 *  Only "the scope id is filled" is checked here; the backend validates the rest and its
 *  answer is shown as the true outcome. */
export function ScopedResetForm({
  busy,
  onCancel,
  onSubmit,
  target
}: {
  busy: boolean
  onCancel: () => void
  onSubmit: (request: MemoryResetRequest) => void
  target: 'memory' | 'user'
}) {
  const { t } = useI18n()
  const mm = t.commandCenter.maintenance
  const [kind, setKind] = useState<(typeof SCOPE_KINDS)[number]>('repository')
  const [scopeId, setScopeId] = useState('')
  const [orgId, setOrgId] = useState('')
  const [projectId, setProjectId] = useState('')
  const [repoId, setRepoId] = useState('')
  const id = scopeId.trim()

  const submit = () => {
    const context = { org_id: orgId.trim(), project_id: projectId.trim(), repo_id: repoId.trim() }

    onSubmit({
      target,
      scopes: [`${kind}:${id}`],
      ...Object.fromEntries(Object.entries(context).filter(([, value]) => value))
    })
  }

  return (
    <div className="mb-2 flex flex-col gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3">
      <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {mm.scopeHint}
      </div>
      <label className="flex items-center gap-2 text-[length:var(--conversation-caption-font-size)]">
        <span className="w-28 shrink-0">{mm.scopeKind}</span>
        <select
          className="h-7 rounded-md border border-(--ui-stroke-tertiary) bg-transparent px-2 text-xs"
          onChange={event => setKind(event.target.value as (typeof SCOPE_KINDS)[number])}
          value={kind}
        >
          {SCOPE_KINDS.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
      <ScopeField label={mm.scopeId} onChange={setScopeId} value={scopeId} />
      <ScopeField label={mm.orgIdField} onChange={setOrgId} value={orgId} />
      <ScopeField label={mm.projectIdField} onChange={setProjectId} value={projectId} />
      <ScopeField label={mm.repoIdField} onChange={setRepoId} value={repoId} />
      <div className="flex justify-end gap-1.5">
        <Button disabled={busy} onClick={onCancel} size="xs" variant="text">
          {mm.resetCancel}
        </Button>
        <Button disabled={busy || !id} onClick={submit} size="xs" variant="textStrong">
          {mm.requestReset}
        </Button>
      </div>
    </div>
  )
}

function ScopeField({ label, onChange, value }: { label: string; onChange: (next: string) => void; value: string }) {
  return (
    <label className="flex items-center gap-2 text-[length:var(--conversation-caption-font-size)]">
      <span className="w-28 shrink-0">{label}</span>
      <Input onChange={event => onChange(event.target.value)} value={value} />
    </label>
  )
}
