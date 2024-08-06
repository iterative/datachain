Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `Apache 2.0 license`_ and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_
- `Code of Conduct`_

.. _Apache 2.0 license: https://opensource.org/licenses/Apache-2.0
.. _Source Code: https://github.com/iterative/datachain
.. _Documentation: https://docs.dvc.ai/datachain
.. _Issue Tracker: https://github.com/iterative/datachain/issues

How to report a bug
-------------------

Report bugs on the `Issue Tracker`_.

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.


How to request a feature
------------------------

Request features on the `Issue Tracker`_.


How to set up your development environment
------------------------------------------

You need Python 3.8+ and the following tools:

- Nox_

Install the package with development requirements:

.. code:: console

   $ pip install nox

.. _Nox: https://nox.thea.codes/


How to test the project
-----------------------

Run the full test suite:

.. code:: console

   $ nox

List the available Nox sessions:

.. code:: console

   $ nox --list-sessions

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

.. code:: console

   $ nox --session=tests

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/


Build documentation
-------------------

If you've made any changes to the documentation (including changes to function signatures,
class definitions, or docstrings that will appear in the API documentation),
make sure it builds successfully.

.. code:: console

   $ nox -s docs

In order to run this locally with hot reload on changes:

.. code:: console

   $ mkdocs serve


How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks, you can invoke a `lint` session in nox:

.. code:: console

   $ nox -s lint

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

.. _pull request: https://github.com/iterative/datachain/pulls
.. github-only
.. _Code of Conduct: CODE_OF_CONDUCT.rst
