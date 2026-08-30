import type { ReadinessState } from './types'

const STATES: readonly string[] = ['loading', 'ready', 'failed'] satisfies ReadinessState[]

/** Narrow an untrusted body to the readiness vocabulary the backend speaks. */
export function isReadinessState(value: unknown): value is ReadinessState {
  return typeof value === 'string' && STATES.includes(value)
}
