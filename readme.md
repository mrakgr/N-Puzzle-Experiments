Repo for the N Puzzle experiments.

3/24/2016: Currently building up towards the [pattern database](https://heuristicswiki.wikispaces.com/pattern+database) approach. After that I indend to bring in [DiffSharp](http://diffsharp.github.io/DiffSharp/).

3/25/2016: Well, the buildup did not last long. Implementing the pattern database was easier than I'd though it would be. The test cases from the Coursera Algorithms I assignment trolled the life out of me though. I thought my past solvers were horrible, but it turns out all the 4x4 puzzle files except the one marked "unsolvable" are in fact unsolvable. What the hell?

In the project directory, the "n_puzzle_pattern_database_v3.fs" will attempt to load the patterns from file and if they aren't present will call the pattern builder to calculate the value functions themselves which takes about half an hour or so. The fringe and the corner patterns themselves take 0.5Gb each, while the two smaller patterns are 100th of that size.

Of the test cases, most take less than a second with some taking up to a minute. On my PC it takes 13m to run through all 100 of them.

Now comes the interesting part. For the first time since I started programming a year and a month ago, I now have the skills and the tools to actually make neural nets do something useful, which is compress those huge 500Mb value function files to something manageable.

I am really curious how well it would go. I have no idea what sort of error rate or compression ration can even be expected at all. How will adding layers affect things? How will I transform the architecture gradually? There is a wide world out there waiting for me.