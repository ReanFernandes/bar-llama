# System prompts

This folder contains all the possible formats of the prompts that will serve as the system prompt to the model. Based on the following, one of these files will be selected using the naming conventions:
<ol>
    <li> <b>Response Format</b> : JSON, Markdown or Numbered list </li>
    <li> <b>Response Type </b> : Fact first or Answer first </li>
    <li> <b>Explanation type </b> : Structured or unstructured Explanation </li> 
</ol>

This forms the first block of the prompt that will be presented to the model, with the example appended in case of a few shot prompt, followed by the question text. 
