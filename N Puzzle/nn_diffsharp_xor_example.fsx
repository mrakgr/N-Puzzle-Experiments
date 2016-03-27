// DiffSharp XOR example.
// It has really been a while since I last did neural nets - I seem to have forgotten quite a bit despite it being only a few months.
// I am starting all over from the basics here.

// ...When I first tried DiffSharp it had a back in the backprop phase and I found that bug and fixed it, eventually making the Spiral library in the
// process.

#if INTERACTIVE
#r "../packages/DiffSharp.0.7.7/lib/net46/DiffSharp.dll"
#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#endif

open System
open DiffSharp.AD.Float32
open DiffSharp.Util

open FSharp.Charting

let rng = Random()

// A layer of neurons
type Layer =
    {mutable W:DM  // Weight matrix
     mutable b:DV  // Bias vector
     a:DM->DM}     // Activation function

// A feedforward network of neuron layers
type Network =
    {layers:Layer[]} // The layers forming this network

let runLayer (x:DM) (l:Layer) =
    l.W * x + l.b |> l.a

let runNetwork (x:DM) (n:Network) =
    Array.fold runLayer x n.layers

let createNetwork (l:int[]) =
    {layers = Array.init (l.Length - 1) (fun i ->
        let l1 = l.[i + 1]
        let l2 = l.[i]
        let s = l1+l2 |> float |> sqrt
        {W = DM.init l1 l2  (fun _ _ -> (-0.5 + rng.NextDouble()) / s |> float32 )
         b = DV.init l1 (fun _ -> (-0.5 + rng.NextDouble()) / s |> float32 )
         a = sigmoid })}

let cross_entropy_cost (inputs:DM) (targets:DM) =
    ((targets .* (DM.Log inputs) + (1.0f-targets) .* DM.Log (1.0f-inputs)) |> DM.Sum) / (-inputs.Cols)

// Backpropagation with SGD and minibatches
// n: network
// eta: learning rate
// epochs: number of training epochs
// mbsize: minibatch size
// loss: loss function
// x: training input matrix
// y: training target matrix
let backprop (n:Network) (eta:float32) epochs mbsize loss (x:DM) (y:DM) =
    let i = DiffSharp.Util.GlobalTagger.Next
    let mutable b = 0
    let batches = x.Cols / mbsize
    [|
    for j=1 to epochs do
        b <- 0
        while b < batches do
            let mbX = x.[*, (b * mbsize)..((b + 1) * mbsize - 1)]
            let mbY = y.[*, (b * mbsize)..((b + 1) * mbsize - 1)]

            for l in n.layers do
                l.W <- l.W |> makeReverse i
                l.b <- l.b |> makeReverse i

            let L:D = loss (runNetwork mbX n) mbY
            L |> reverseProp (D 1.0f)

            for l in n.layers do
                l.W <- primal (l.W.P - eta * l.W.A)
                l.b <- primal (l.b.P - eta * l.b.A)

            printfn "Epoch %i, minibatch %i, loss %f" j b (float32 L)
            b <- b + 1
            yield float32 L
        |]

let XORx = 
    [|0.; 0.; 0.; 1.; 1.; 0. ;1.; 1.|]
    |> Array.map (float32 >> D)
    |> DM.ofArray 4
    |> DM.transpose
             
let XORy = 
    [|0.; 1.; 1.; 0.|]
    |> Array.map (float32 >> D)
    |> DM.ofArray 4
    |> DM.transpose

// 2 inputs, 3 neurons in a hidden layer, 1 neuron in the output layer
let net3 = createNetwork [|2; 15; 1|]

// Train
let train3 = backprop net3 4.0f 1000 4 cross_entropy_cost XORx XORy

// Plot the error during training
Chart.Line train3
|> Chart.Show