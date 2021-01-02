# Machine Learning Engineer Nanodegree
## Capstone Proposal
Adelmo M. A. Filho  
January 1st, 2020

<br>
<h1 align="center">Predicting the Brazilian stock market index through recurrent neural networks</h1>

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs

The datasets are provided the python package [Yahooquery](https://yahooquery.dpguthrie.com/) which works as a wrapper for an unofficial [Yahoo Finance](https://finance.yahoo.com/) API. Data used on this project was obtained for free, there was no need for a Yahoo Finance premium subscription.

The `history` method from `Ticker` class of Yahooquery package allows to retrive daily data about stock markets. The following code displays how to get historical data for the Bovespa Index, also a sample of this data is presented.

```
from yahooquery import Ticker
ibov = Ticker(symbols = "^BVSP")
ibov.history(period="max")
``` 

| symbol | date       | open    | close   | low     | high    | volume     | adjclose |
|--------|------------|---------|---------|---------|---------|------------|----------|
| ^BVSP  | 2020-04-08 | 76335.0 | 78625.0 | 76115.0 | 79058.0 | 10206300.0 | 78625.0  |
| ^BVSP  | 2020-04-07 | 74078.0 | 76358.0 | 74078.0 | 79855.0 | 11286500.0 | 76358.0  |
| ^BVSP  | 2020-04-06 | 69556.0 | 74073.0 | 69556.0 | 75260.0 | 9685400.0  | 74073.0  |
| ^BVSP  | 2020-04-03 | 72241.0 | 69538.0 | 67802.0 | 72241.0 | 10411300.0 | 69538.0  |
| ^BVSP  | 2020-04-02 | 70969.0 | 72253.0 | 70957.0 | 73861.0 | 10540200.0 | 72253.0  |

Not only Bovespa Index data is expected to be used on this project, but also historical data from the main stocks that represents its portfolio. The following table presents the main stocks that compose Bovespa Index and their global participation on the portfolio.

| Ticker | Company                              | IBOVESPA Participation |
|--------|--------------------------------------|------------------------|
| ITUB4  | Itaú Unibanco Holding S.A.           | 10,50%                 |
| BBDC4  | Banco Bradesco S.A.                  | 9,12%                  |
| VALE3  | Vale S.A.                            | 8,59%                  |
| PETR4  | Petróleo Brasileiro S.A. - Petrobras | 7,06%                  |
| PETR3  | Petróleo Brasileiro S.A. - Petrobras | 5,14%                  |
| ABEV3  | Ambev S.A.                           | 5,14%                  |
| BBAS3  | Banco do Brasil S.A.                 | 4,47%                  |
| B3SA3  | B3 S.A. - Brasil, Bolsa, Balcão      | 4,15%                  |
| ITSA4  | Itaúsa - Investimentos Itaú S.A.     | 3,86%                  |


### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?