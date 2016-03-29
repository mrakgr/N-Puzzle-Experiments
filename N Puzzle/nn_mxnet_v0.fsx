// First time trying the R type provider. I am not sure whether I will be able to get to the MXNet library from F#.

#load "../packages/RProvider.1.1.15/RProvider.fsx"

open System
open RDotNet
open RProvider
open RProvider.graphics

// Edit: Nevermind, I'll just add convolutional functions to my own library. 
// Is it just me or is there something about getting other DL libraries to work that drives people to madness?
// Do they make them as hard as possible to install on purpose or something?

