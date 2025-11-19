# CLAUDE REQUIREMENTS

## Overview

I want to refactor the fastapi_app to make it simpler and more effective. The main goal is to completely remove the 
LLM model that is an overkill for the purpose. The idea is to use only the vector store to identify the scenario that
the user wants to use and return the found scenario or nothing if the relevance score is lower than a predefined threshold.

## Steps 

- change the document indexer to work differently:
  - checkout the website repository
  - scan all the .md files in the /docs folder of the website repository. Documents that need to be indexed will contain a tag 
    "<krkn-hub-scenario id="">" the content of this tag is the *only* content that needs to be indexed and the id of the tag 
    is the exact scenario id that needs to be returned by the fast API and nothing else.
- when the user enters a query the vector store needs to be searched and the most relevant document id must be returned if the
  relevance is under a certain level nothing must be returned.
    