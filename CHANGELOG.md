# Changelog

## [0.2.0](https://github.com/dwest1507/nietzsche-chat/compare/nietzsche-chat-v0.1.0...nietzsche-chat-v0.2.0) (2026-08-31)


### Features

* add a setup wizard for the manual platform steps ([714c649](https://github.com/dwest1507/nietzsche-chat/commit/714c6495b062ce32855dcaad6dc1244563c66972)), closes [#12](https://github.com/dwest1507/nietzsche-chat/issues/12)
* distinguish provider quota from generic failures on the stream ([d61dae3](https://github.com/dwest1507/nietzsche-chat/commit/d61dae3bbf210a629187cb68814014153dc6cb1c)), closes [#8](https://github.com/dwest1507/nietzsche-chat/issues/8)
* hold a question sent while the backend wakes, then dispatch it ([8924e83](https://github.com/dwest1507/nietzsche-chat/commit/8924e8344d2ee7352b6fb377c27fbf435b36e673)), closes [#7](https://github.com/dwest1507/nietzsche-chat/issues/7)
* rate limit per visitor on the forwarded client address ([16f1fa9](https://github.com/dwest1507/nietzsche-chat/commit/16f1fa98248d4a3153c1cfd5a7505a62a7028337)), closes [#5](https://github.com/dwest1507/nietzsche-chat/issues/5)
* report backend errors to Sentry ([1cab5d2](https://github.com/dwest1507/nietzsche-chat/commit/1cab5d278b8a971e1c9766eecf0952884bb781ab)), closes [#10](https://github.com/dwest1507/nietzsche-chat/issues/10)
* report readiness separately from liveness and surface a waking state ([10d84f7](https://github.com/dwest1507/nietzsche-chat/commit/10d84f7513a67ed798ed3d019ff47a8b7e49a3f3)), closes [#6](https://github.com/dwest1507/nietzsche-chat/issues/6)
* require a shared secret on the chat endpoint ([105b043](https://github.com/dwest1507/nietzsche-chat/commit/105b0432b7fe30559b129be19c38f7b0eb531cdc)), closes [#4](https://github.com/dwest1507/nietzsche-chat/issues/4)


### Bug Fixes

* **ratelimit:** key the limiter on the address the visitor cannot write ([691dd20](https://github.com/dwest1507/nietzsche-chat/commit/691dd20c70f85c6c9e8d88b0e5c6cb76e9a92a6f))
* **readiness:** bound the wake window and re-check a stale ready ([08f1b8c](https://github.com/dwest1507/nietzsche-chat/commit/08f1b8c7a8fcde09186f73946176cb1bbc63d3ca))
* **readiness:** treat a cold-start 5xx as waking, not as terminal failure ([7d79806](https://github.com/dwest1507/nietzsche-chat/commit/7d798062e69bb7b0bc9920d1a511d866f727d125))
* **startup:** retry a failed warm-up instead of latching failed forever ([f0fa4e7](https://github.com/dwest1507/nietzsche-chat/commit/f0fa4e79ae6489effafb82318871f87dbb2bbe48))

## 0.1.0 (2026-08-30)


### Features

* automate versioning and releases with Release Please ([#9](https://github.com/dwest1507/nietzsche-chat/issues/9)) ([0447a16](https://github.com/dwest1507/nietzsche-chat/commit/0447a16637910bd7b8d0b43e61b261ba9d35ac8d))
