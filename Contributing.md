# Contribution Guidelines

## Steps to Contributing

Here are the basic steps to get started with your first contribution. Please reach out with any questions.
1. [Fork the repo](https://help.github.com/articles/fork-a-repo/) so you can make and test local changes.
1. Create a new branch for the issue. We suggest prefixing the branch with your username and then the master branch name: (e.g. username/bertonazureml)
1. Make code changes.
1. We use [pre-commit](https://pre-commit.com/) package to run our pre-commit hooks. We use black formatter and flake8 linting on each commit. In order to set up pre-commit on your machine, follow the steps here, please note that you only need to run these steps the first time you use pre-commit for this project.
   
   * Update your conda environment, pre-commit is part of the yaml file or just do    
   ```
    $ pip install pre-commit
   ```    
   * Set up pre-commit by running following command, this will put pre-commit under your .git/hooks directory. 
   ```
   $ pre-commit install
   ```
   ```
   $ git commit -m "message" 
   ```
   * Each time you commit, git will run the pre-commit hooks (black and flake8 for now) on any python files that are getting committed and are part of the git index.  If black modifies/formats the file, or if flake8 finds any linting errors, the commit will not succeed. You will need to stage the file again if black changed the file, or fix the issues identified by flake8 and and stage it again.

   * To run pre-commit on all files just run
   ```
   $ pre-commit run --all-files
   ```
1. Create a pull request against <b>bertonazureml</b> branch.

Note: We use the bertonazureml branch to land all new features, so please remember to create the Pull Request against staging. 

Once the features included in a milestone are complete we will merge staging into master and make a release.