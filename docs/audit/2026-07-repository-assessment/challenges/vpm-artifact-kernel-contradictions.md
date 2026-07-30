# Contradictions and Terminology Issues

README.md uses broad phrases including "Visual AI Computing", "intelligence", and "read as a sign". The bounded caveats in the same file and docs/claims-audit.md are important; without them, the implementation supports deterministic finite artifact lookup, not general visual intelligence or reasoning. Narrow wording: "identified finite score/policy artifacts with deterministic lookup and traceability."

docs/claims-audit.md accurately preserves several negative results, including PNG not being self-describing and learned visual-address claims being unsupported under corrected calibration. That is a positive documentation practice.

demos/catalog.json marks VPM and Lua demos with evidence_state "defined". The demos import production examples and packages, but "defined" is weaker than "validated" and does not itself prove execution. The build script/catalogue should be read as demo metadata, not independent evidence.

Rendering helpers produce grayscale PNG/SVG from normalized fields. They do not embed self-describing artifact metadata. Any phrase implying PNG itself is a self-describing VPM would be overstated.

"Compiled" is accurate for the in-memory plan and generated Lua table, but it means deterministic materialization of lookup data, not optimization, bytecode compilation, or real-time/embedded performance proof.
