# MLP

![[Pasted image 20250409105246.png]]

Core idea is usage of learned embeddings $C$
It is a matrix of size vocab_size $\times$ dim.
As input to the net we pass indices to plug out embedding from row C for particular element (char or word, depends). We concatenate so it's 3$\times$dim inputs (3 - context window size) & everything that comes after is a simple multiclass classification feed-forward net.

Mapping stoi; load dataset in dot-filled trigrams. X trigram -> y character
shift context window; dot padding; Create embedding matrix 2D;

> Intuition: Input OHE representation of chars. Weights - C matrix (embedding)

However, we'll use indices.
PyTorch supports multiple indexing via lists
```python
x = torch.tensor([1, 2, 3])
x[[1, 2]]  # >>> torch.tensor([2, 3])
```
Since X stores arrays of numbers (triplets) we can index this way `C[X]`
It results in `[m, 3, dim]` shape of the output (embedding for each char in each triplet)
<p style="color: red;"> WHY? </p>
<p style="color: lime">I think It's like X is array of triplets. Its first item is [0, 0, 0].
So C[X[0]] is nothing, but [emb[0], emb[0], emb[0]] resulting in [emb0, emb0, emb0] where emb0 is embedding of 0 with shape (2,)</p>

To concatenate these embeddings (lower dimension, so that rows consist of embedded scalars)
`[32, 3, 2]` -> `[32, 6]`

(NOTE: Read Ezyang's blog about **PyTorch internals**)

> Cool stuff:
>>Pytorch has two methods `.view()` and `.reshape()` with similar purpose.
>>However, the way they perform is super different.
>>`.reshape()` rearranges objects in memory in such a way as it was told to do
>>while `.view()` doesn't rearrange something. He just changes his perspective on how to represent and return the data
>>So yep, the difference between these two is revealed in their names.
>>Btw: `.view()` is more efficient, yet can't be used after some operations (Tranpose etc.)

Speaking of loss function, instead of reinventing the wheel we can use `F.cross_entropy()` loss, since it's tuned and tested by the devs.

> Cool stuff:
>> loss in such NLP tasks can't be zero, since given the same starting context "..." we have many many words starting with various characters. It's like "..." -> "e" for word emma, but at the same time "..." -> "m" for maxwell.

