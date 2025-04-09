This directory contains all thats needed to format the training data set and augment it with the explanation structure that I created. I am using LLaMa 3.1 70B Q6_K quantised model, which we are self hosting. The code has been modified from a boilerplate example to fit this purpose. Theres not much wiggle room here, maybe if the prompt needs to be changed it could, but since the inference is expensive im not really keen on trying different types. JSON schema is enforced to ensure fewer parsing errors, hopefully the quantisation doesnt hurt the model performance too much. 


The api calls are in the OpenAI api format, so theoretically one could replace their api key and perform the same distillation, which we did with our model. 
