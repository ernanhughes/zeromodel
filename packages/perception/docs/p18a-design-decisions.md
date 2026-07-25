# P18A Design Decisions

1. **Preserve both observations.** The transition artifact supplements the before and after Source VPMs; it does not replace them.
2. **Reuse P4A fields.** P18A does not invent a second spatial addressing system.
3. **Reuse P6 annotations.** Known object identities bind to fields without influencing measurement.
4. **Measure before interpretation.** Absolute change, signed change, and threshold-crossing fraction are recorded before later conformance logic.
5. **Keep the first slice non-causal.** Recurrent and expected transition reasoning belongs in P18B and P18C.
6. **Make the rendering evidence-bearing.** The PNG digest and complete measurement payload are included in the transition identity.
