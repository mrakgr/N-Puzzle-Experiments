// Tried softmax as well. It is roughly on par with logistic regression.

#if INTERACTIVE
#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#load "n_puzzle_3x3_builder.fsx"
#load "spiral_conv.fsx"
#endif
open PatternBuilder3x3
open SpiralV2

open System
open System.Collections.Generic
open FSharp.Charting

let inputs_targets =
    value_function
    |> Array.map (
        fun (l,r) ->
        d4M.createConstant(l.Length/81,81,1,1,l),
        d4M.createConstant(r.Length/32,32,1,1,r)
        )

let layers = 
    [|
    FeedforwardLayer.create (81,32,1,1) relu :> INNet
    ResidualFeedforwardLayer.create (32,32,1,1) relu relu :> INNet
    ResidualFeedforwardLayer.create (32,32,1,1) relu relu :> INNet
    ResidualFeedforwardLayer.create (32,32,1,1) relu relu :> INNet
    ResidualFeedforwardLayer.create (32,32,1,1) relu relu :> INNet

    ResidualFeedforwardLayer.create (32,32,1,1) relu relu :> INNet
    ResidualFeedforwardLayer.create (32,32,1,1) relu relu :> INNet
    ResidualFeedforwardLayer.create (32,32,1,1) relu relu :> INNet

    FeedforwardLayer.create (32,32,1,1) clipped_softmax :> INNet
    |]

81*32+32*32*14+32*32 // number of parameters

let training_loop label data i =
    let increase_i_and_get_factor () = 
        let i' = !i
        i := i'+1
        1.0/(1.0 + float i')
    layers
    |> Array.fold (fun x layer -> layer.train x increase_i_and_get_factor) data
    |> fun x -> lazy get_accuracy label x, cross_entropy_cost label x

let test learning_rate num_iters =
    [|
    let c = ref 0
    for i=1 to num_iters do
        let mutable er = 0.0f
        let mutable acc = 0.0f
        for input, label in inputs_targets do
            let acc',r = training_loop label input c // Forward step
            er <- er + !r.P
            acc <- acc'.Value + acc
            ObjectPool.Reset() // Resets all the adjoints from the top of the pointer in the object pool along with the pointers.
            layers |> Array.iter (fun x -> x.ResetAdjoints())
            //printfn "Squared error cost on the minibatch is %f at batch %i" !r.P j

            if !r.P |> Single.IsNaN then failwith "Nan!"

            r.A := 1.0f // Loads the 1.0f at the top
            while tape.Count > 0 do tape.Pop() |> fun x -> x() // The backpropagation step
            layers |> Array.iter (fun x -> x.SGD learning_rate) // Stochastic gradient descent.
        printfn "Error is %f. Accuracy is %i/%i on iteration %i." er (int acc) number_of_examples i
        yield int acc
    |]

#time
test 0.05f 95
//|> Chart.Line
//|> Chart.Show
#time
