# EnDeGen: Encoder-Decoder Text Generation with RNNs

EnDeGen is a project that focuses on generating text using the Encoder-Decoder architecture with Recurrent Neural Networks (RNNs). It leverages the power of deep learning to predict and generate text sequences based on given input data.

## Overview

EnDeGen is designed to generate text by learning patterns from a given dataset and predicting the next characters in a sequence. It utilizes the Encoder-Decoder model, which consists of two components:

* The **CharacterLanguageModel** class is responsible for `encoding` the input sequence. It takes the input characters, embeds them using an embedding layer, and passes them through a GRU (Gated Recurrent Unit) layer. The GRU layer processes the input sequence and returns the hidden states. Finally, the output of the GRU layer is passed through a dense layer to obtain the encoded representation of the input sequence.

* The **OneStep** class is responsible for `decoding` the encoded representation and generating the next character in the sequence. It takes the encoded representation, feeds it into the language model, and obtains the predicted logits for the next character. It then applies a `temperature` parameter to control the randomness of the predictions and applies a mask to prevent the generation of unknown characters. Finally, it samples from the predicted logits to generate the next character and returns it along with the model state.



## Example Result 
**[After training for 30 epochs, the model was able to generate the following text]**

All: How nigh should do boys where no harm to keep
Than younger conference was a point of your
Boint powers. What, will our princely father York?
Upon my love, think'st thou that I am little
That go to chastity of our intone humour
will church the mighty look od all in 

Bolingbroke:
Be it so I have the let morn of your
having breath? Of high and lowly me!
So, fellow, get me heaven, out o'er at Sicilia
much in placent, all the prisoner to do fight:
I do remember this be this: he's very little
That wound this way how touch'd Slones, where you
will have consumenes to yourself. You talk-worng, would wed both
He thinks me down to such a party instinct,
Travelling to practise himself be in love:
I'll blush your honour.

ISABELLA:
Which is too careless it and me?
My grave be loss, that you hither hath a hand,
And shows good words will maid to see thee dead,
How doth the queen of Naples him.
But let it serve to take our foes.

ANTONIO:
I beseech your parish close by Caling home,
That his bold hand.


## Contact
For any questions or inquiries about the project, please reach out to [LinkedIn](https://www.linkedin.com/in/mohres)

Happy text generation with EnDeGen!