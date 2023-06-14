# Instructions for running tests for GTST

The requirements for running test for GTST is listed in the `requirements_dev.txt`. The test involve testing if kernels as positive semi definite and weather the kernels, test statistic, and permutation method for the p-value are able to reject the null when the null is "extremely" false.

## Testing with PyTest

Once the required packages have been installed, the tests can be performed by running the command 

```
pytest
```

in the root folder. Will take around 15 minutes. This will generate a coverage report which can be found in the `htmlcov` directory. To view it run

```
cd htmlcov && python -m http.server
```

and open the localhost link (something like http://localhost:8000/) in a browser.

## Testing with Tox

To test GTST in a clean environment for all python versions from 3.7-3.10, we use Tox. This can be achieved by running 

```
tox
```

in the root directory. Note that this takes significantly longer to run, so is best performed as a final check. 

## Testing with GitHub actions

Whenever code is pushed to the remote repository, the Tox test suite is automatically run using GitHub actions. To investigate this process, consult the file found at `.github/workflows/tests.yml`.