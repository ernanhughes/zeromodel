1. ✅  python .\scripts\bump_version.py patch
2. ✅ `python -m build`
3. ✅ `python -m twine check dist/*`
4. ✅ (optional) Upload to TestPyPI, quick smoke test
5. ✅ `python -m twine upload dist/*`
6. ✅ `pip install zeromodel==<new_version>` in a clean env to confirm
7. ✅ Tag & push

