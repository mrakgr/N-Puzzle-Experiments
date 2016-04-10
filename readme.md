Repo for the N Puzzle experiments.

3/24/2016: Currently building up towards the [pattern database](https://heuristicswiki.wikispaces.com/pattern+database) approach. After that I indend to bring in [DiffSharp](http://diffsharp.github.io/DiffSharp/).

3/25/2016: Well, the buildup did not last long. Implementing the pattern database was easier than I'd though it would be. The test cases from the Coursera Algorithms I assignment trolled the life out of me though. I thought my past solvers were horrible, but it turns out all the 4x4 puzzle files except the one marked "unsolvable" are in fact unsolvable. What the hell?

In the project directory, the "n_puzzle_pattern_database_v3.fs" will attempt to load the patterns from file and if they aren't present will call the pattern builder to calculate the value functions themselves which takes about half an hour or so. The fringe and the corner patterns themselves take 0.5Gb each, while the two smaller patterns are 100th of that size.

Of the test cases, most take less than a second with some taking up to a minute. On my PC it takes 13m to run through all 100 of them.

Now comes the interesting part. For the first time since I started programming a year and a month ago, I now have the skills and the tools to actually make neural nets do something useful, which is compress those huge 500Mb value function files to something manageable.

I am really curious how well it would go. I have no idea what sort of error rate or compression ration can even be expected at all. How will adding layers affect things? How will I transform the architecture gradually? There is a wide world out there waiting for me.

3/30/2016: Where did the last five days ago? I did all sorts of experimenting and managed to compress 1/2000th of the 5MB pattern at a 90% accuracy. At first I thought it was due to unsupervised learning, but it seems the net learns much faster with smaller minibatches.

5Mb seems small at a first glance and I did not expect it to be hard, but it turns out once I unpack the scalars into one-hot encodings and turn them into floats, the 5MB pattern turns into 7Gb. Yikes!

I also seem to be making progress at a really snail's pace. What I am going to do is switch to the 3x3 puzzle and try to solve that. The total number of states for the 3x3 puzzle is only around 180k which is 30 times less than even the "small" 4x4 pattern. Before I do that, I'll first resume work on the [Spiral library](https://github.com/mrakgr/Spiral-V2) and finally do it the right way, along with the convolutional functions.

This will only take me a week or two hopefully, but after I do this I will be ready to deal with the 3x3 puzzle. After that I'll think about my next move.

4/8/2016: Ready to start again.

4/10/2016: Done.

Before I take the next step, let me summarize what I've done up until now.

First of, I've tested whether a wide, shallow net of 4M parameters can memorize the entire dataset. That proved to be positive. After that I've tried much deeper and narrower nets. Of about 8-15k parameters and 10-22 layers depending on the net. Roughly, it manages to get 50% accuracy on half of the dataset, but then the optimization problem becomes particularly difficult and it continues improving in a very slow, grinding manner.

I cannot tell whether if I let it train for much longer than 300 iteration whether it would break out to a much higher plateau. The cost function as it moves down is very smooth and regular.

For a 15 layer residual layer, quadrupling the number of parameters, only increases its accuracy from 100k to 115k (out of 181k) after 100 iterations.

Out of the three potential enhancements -: Batch Normalization, Residual Learning and Stochastic Depth – only residual learning in particular gave any improvement, allowing me to train much deeper nets without them blowing up. The effect of stochastic depth seems to be marginal and batch normalization made things much worse in particular. The effect of batch normalization in particular is surprising given its stellar performance on general machine learning tasks. Maybe if I tried dropout, I would observe the same thing?

As I stand back and reflect on what I have done, I am sure that the question of why batch normalization works so poorly, why increasing the width has such poor scaling and most importantly, the question of why the cost and accuracy curve behave the way they do have some profound implications, but at the moment they are beyond my imagination and my ability to answer in much the same way the 10k parameter network has difficulty pushing through the dataset to some true insight lying beyond.

N puzzle – a NP Hard problem – seems to have a dual structure. Not in terms of invariance, but it seems it has an easy and a hard part. On the 181k sized dataset, it seems trivial for the network to eat away the first 80-100k or so, but that other hard is way too much for it to swallow.

Extrapolating by induction, maybe, just maybe, what if the 4x4, 5x5 and beyond have that same structure?

It does not have to be 50/50. Just how much would having an understanding even a single % of a 1e^100 sized state space truly be worth? And how much could just a tiny piece of generalization cut into the enormity of NP Completeness?

A question worth asking might be how small can I make the nets?

...With 8 units in each hidden layer, I get 77K correct after 100 iterations. This is remarkable with a net of only 1800 parameters.

I guess that alone demonstrates where the power of neural nets truly lies – in the easy part.

And with that I will bring the N puzzle adventure to a close. I am satisfied with this. The 3x3 example was informative enough.

I had wanted to try 4x4 and more, but starvation looms on the horizon and it is time to put my skills to the test. Finally. I want a piece of that low hanging fruit.
