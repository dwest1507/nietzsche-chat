import '@testing-library/jest-dom'
import { vi } from 'vitest'

// jsdom does not implement scrollIntoView (used by ChatShell autoscroll)
window.HTMLElement.prototype.scrollIntoView = vi.fn()
