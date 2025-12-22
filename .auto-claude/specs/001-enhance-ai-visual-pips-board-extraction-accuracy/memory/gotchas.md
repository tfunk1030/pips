# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-22 05:43]
Running bare `npx tsc --noEmit` on pips-solver produces many pre-existing TypeScript errors due to Expo/React Native type definition conflicts. This is expected - the project compiles correctly when run through Expo bundler.

_Context: subtask-4-1 verification - TypeScript checking in pips-solver_
