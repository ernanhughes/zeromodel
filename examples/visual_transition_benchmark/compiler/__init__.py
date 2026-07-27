"""Evidence Contract Compiler: compiles a declared visual evidence requirement
into an explicit, inspectable representation plan (region, field resolution,
aggregation, decoder, comparison), instead of hand-selecting one per domain.

This is a bounded benchmark instrument, not a new perception stage. Nothing
here is imported by ``zeromodel.perception``; it only imports from it (via
``compilation.field_schema_compiler``, reused unchanged from the cross-domain
experiment).
"""
