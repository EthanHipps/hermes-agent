"""Writes the setup facts agreed during onboarding into the default profile's user memory."""
import json

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.profiles import get_profile_dir
from tools.memory_tool import load_on_disk_store, memory_tool

_RESULT = {'profile': 'default', 'target': 'user'}


def _default_profile_is_provider_managed() -> bool:
    """§9.7 L1605: the default profile's requested mode, through the one pipeline archives use (C6b-10)."""
    from hermes_cli.backup_memory import home_disposition
    return home_disposition(get_profile_dir('default')) is not None


def _remember_provider_managed(content: str) -> dict:
    """The facts as one ``target:user`` add through MemoryService, under the default profile's principal-global
    administrative identity (§9.7 L1605; contract C6b-5). A user-profile change always needs approval and
    nobody can answer a prompt here, so it waits in the pending memory approvals (C6b-9): ``saved`` stays
    False and ``pending_id`` names it. No ``USER.md`` is read or written."""
    from agent.memory_service.admin import (ADMIN_FAILURES, AdminContext, admin_approval_channel, admin_service,
                                            outcome_payload, unavailable_payload)
    from agent.memory_service.mutation import MutationStatus, run_curated_mutation
    from hermes_cli.backup_memory import home_memory_section
    from tools.memory_tool_curated import ConsolidationBudget, NativeCall, plan_native_call

    raw = {'memory': dict(home_memory_section(get_profile_dir('default')))}
    call, budget = NativeCall(action='add', content=content, old_text=None, operations=None), ConsolidationBudget()
    try:
        approval = admin_approval_channel(raw, AdminContext(), origin='onboarding')
        with admin_service(raw, context=AdminContext()) as service:
            outcome = run_curated_mutation(service, 'user', lambda snapshot: plan_native_call(snapshot, call, budget),
                                           actor_kind='human', initiating_surface='onboarding', approval=approval)
    except ADMIN_FAILURES as exc:
        raise ValueError(unavailable_payload(exc)['message']) from None
    if outcome.status is MutationStatus.PENDING_APPROVAL:
        return {'saved': False, 'pending_id': outcome.pending_id, **_RESULT,
                'message': outcome_payload(outcome)['message']}
    if outcome.status is MutationStatus.COMMITTED:
        return {'saved': True, **_RESULT}
    response = outcome.short_circuit.response if outcome.short_circuit else outcome_payload(outcome)
    if outcome.status is MutationStatus.SHORT_CIRCUIT and response.get('success'):
        return {'saved': True, **_RESULT}              # already remembered
    raise ValueError(response.get('error') or response.get('message') or 'Memory was not saved')


def remember_onboarding(answers: dict) -> dict:
    if not isinstance(answers, dict):
        raise ValueError('Onboarding answers must be an object')
    facts = []
    for key, label in (('name', 'User prefers to be called'), ('context', 'Working on'),
                       ('theme', 'Desktop theme'), ('accent', 'Desktop accent'), ('layout', 'Desktop layout')):
        value = answers.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'{key} must be text')
        if value and value.strip():
            facts.append(f'{label}: {value.strip()}')
    for key, label in (('focus', 'Focus areas'), ('connectors', 'Tools the user uses (not connection status)')):
        values = answers.get(key, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f'{key} must be a list of text')
        if values := list(dict.fromkeys(value.strip() for value in values if value.strip())):
            facts.append(f'{label}: {", ".join(values)}')
    if not facts:
        return {'saved': True, **_RESULT}
    content = 'Agreed during onboarding:\n' + '\n'.join(facts)
    if len(content) > 2000:
        raise ValueError('Onboarding facts are too long to remember')

    # The entry must land in the 'default' profile directory even when this RPC arrives on the guide's
    # backend or under a custom Hermes home.
    token = set_hermes_home_override(get_profile_dir('default'))
    try:
        if _default_profile_is_provider_managed():
            return _remember_provider_managed(content)
        result = json.loads(memory_tool(action='add', target='user', content=content, store=load_on_disk_store()))
        if not result.get('success') or result.get('staged'):
            raise ValueError(result.get('error') or result.get('message') or 'Memory was not saved')
        # memory_tool can report success without the entry reaching disk, so read it back from a fresh store.
        if content not in load_on_disk_store().user_entries:
            raise ValueError('Could not verify saved onboarding facts')
        return {'saved': True, **_RESULT}
    finally:
        reset_hermes_home_override(token)
