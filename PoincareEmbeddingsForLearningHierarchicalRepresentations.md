This paper considers the problem of learning hierarchical representations for symbolic data structures
such as graphs and language entities.

The main idea is to treat the embedding space as a Poincare Half Space Model.
The paper shows that this indeed leads to a huge performance boost as well as
giving some theoretical motivation for why. 

(Thor) The reason why seems largely
to be that the symbolic data structures considered have complicated
properties (complex systems) which can not be well expressed in on a Euclidean
domain but can be well expressed with a change of geometry.
In particular, here we consider going from a space with zero curvature
(parallel lines don't intersect) with a space with a constant negative
curvature (parallel lines can intersect e.g. the north pole of a sphere).

Introduction:
In recent years there has been a
breakthrough in NLP in using stochastic gradient descent to optimize word
embeddings, as well as using deep learning architectures to learn
representations of words or other language entities.

These two approaches share the property that they are unsupervised learning
techniques. Both of these approaches learn an embedding layer which can be used
for both generative and discriminative tasks (e.g. machine translation/grammar
correction and
semantic analysis/topic classification (note the analog with conditional
generative tasks in vision such as classification/segmentation/human pose). 

It is important to understand the differences between these two approaches of modelling data. 
In deep learning we care about making sequential probabilistic models which compute continuous representations.
Here we are focused in learning a set of weights corresponding to words in a vocabulary. 

In both cases we end up with some Riemannian manifold which (in line with the distributional hypothesis) in which 
vector representations corresponding to similar words are close to each other.

Preliminaries:

Word2Vec

Word2vec is a shallow architecture consisting of a single hidden layer and a
binary classifier, which given the context of a word `w_{True}` learns to 
discriminate `w_{True}` from a set of `k` other words.

Some code taken from a pytorch tutorial in the appendix.


Riemannian Optimization:

Questions to Answer:
    What is linesearch minimization?

    What is a manifold?

    What is a Riemannian manifold?

    What is the retraction operator?

    http://www.cims.nyu.edu/~wirth/optimization.pdf

    What is Riemannian Optimization and what is it's history?

    What does it mean to differentiate with respect to a metric?

    What is the full meaning of being able to 

Related Work:


```python
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!
```
