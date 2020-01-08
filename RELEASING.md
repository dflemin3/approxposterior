1. Update version in approxposterior/__init__.py to, e.g. 0.4

2. Update version in doc/conf.py

3. Create/Make sure CHANGES.md is up to date for the release

4. Commit change and push to master

       git add . -u
       git commit -m "MAINT: bump version to 0.4"
       git push origin master

5. Tag the release:

       git tag -a v0.4 -m "version 0.4 release"
       git push origin v0.4

6. Create the release distribution

       python setup.py sdist bdist_wheel

7. Upload to PyPI via twine

       twine upload dist/*

8. Build and push the docs website:

       cd doc
       make html
       ghp-import -n -m "message" _build/html/
       git push origin gh-pages

9. update version in doc/conf.py

10. add a new changelog entry for the unreleased version

11. Commit change and push to master

       git add . -u
       git commit -m "MAINT: bump version to 0.4.0dev"
       git push origin master
