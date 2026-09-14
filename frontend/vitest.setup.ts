import '@testing-library/jest-dom'
import type { TestingLibraryMatchers } from '@testing-library/jest-dom/matchers'
import { vi } from 'vitest'

// jest-dom registers its matchers on the global `expect` at import time, but it
// only ships type augmentations for jest and for Vitest 4's single-parameter
// `Assertion<T>`. Vitest 5 dropped the global jest namespace bridge and made
// `Assertion` two-parameter, so declare the matchers against `Matchers<R, T>`,
// Vitest's supported extension point. The parameter list must match Vitest's
// own declaration exactly for the interfaces to merge.
declare module 'vitest' {
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface Matchers<
    R extends void | Promise<void> = void | Promise<void>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    T = unknown,
  > extends TestingLibraryMatchers<unknown, R> {}
}

// jsdom does not implement scrollIntoView (used by ChatShell autoscroll)
window.HTMLElement.prototype.scrollIntoView = vi.fn()
