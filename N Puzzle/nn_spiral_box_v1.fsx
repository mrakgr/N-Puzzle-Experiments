// No, even with feedforward nets, I should be able to memorize a single minibatch perfectly.
// It is true that convolutional nets and pretraining would help, but this is not the limit of their ability.

// I have no idea what I will do if I am wrong here.

// I never experienced the need for courage when programming, but I need to make stronger moves here.
// I've been too gutless.

// Edit: No, it works much worse than having vector outputs. I cannot make the cost function budge.
// Before I try anything else, I'll try asking around.

#if INTERACTIVE
#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#load "nn_auxilaries.fs"
#load "spiral.fsx"
#endif
open NNAuxilaries
open Spiral

open System
open System.Collections.Generic
open FSharp.Charting

let _, (inputs_outputs), carr =
    let mapping_function_box_lr =
        function
        | 0uy -> 1uy
        | 5uy -> 2uy
        | 6uy -> 3uy
        | 8uy -> 4uy
        | 9uy -> 5uy
        | 10uy -> 6uy
        | _ -> 0uy
    (mapping_function_box_lr,IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_box_lr.dat"), IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_box_lr.carr"))
    |> fun (f,filename_dat,filename_carr) -> 
        let chunk_size = 128
        (f, IO.File.ReadAllBytes(filename_dat).[0..chunk_size-1], IO.File.ReadAllBytes(filename_carr) |> Array.map int64)
        |> fun (f, value_function, carr) ->
        f,
        (
        let histogram = 
            Dictionary(HashIdentity.Structural)
            |> fun t ->
                for x in value_function do
                    match t.TryGetValue x with
                    | true, v -> t.[x] <- v+1
                    | false, _ -> t.[x] <- 1
                t |> Seq.toArray |> Array.map (fun x -> x.Key, x.Value)
        
        Chart.Bar (histogram, YTitle="Number of Patterns",XTitle="Move Length",Title="Value Function Histogram")
        |> Chart.Show

        let outputs =
            value_function
            |> Array.chunkBySize chunk_size
            |> Array.map ( fun x ->
                x 
                //|> array_decoder 255
                |> Array.map float32
                |> fun x -> DM.makeConstantNode(1,chunk_size,x)
                )

        let inputs = 
            [|0L..value_function.Length-1 |> int64|]
            |> Array.chunkBySize chunk_size
            |> Array.map (fun x ->
                x 
                |> Array.map (multinomial_decoder carr >> array_decoder 7)
                |> Array.concat
                |> fun x -> DM.makeConstantNode(7*16,chunk_size,x)
                )
                
        Array.zip inputs outputs
        ),
        carr

let layers = 
    [|
    FeedforwardLayer.createRandomLayer 3072 112 tanh_
    //FeedforwardLayer.createRandomLayer 3072 3072 tanh_
    FeedforwardLayer.createRandomLayer 1 3072 (clipped_steep_sigmoid 3.0f)
    |]

// This does not actually train it, it just initiates the tree for later training.
let training_loop (data: DM) (targets: DM) (layers: FeedforwardLayer[]) =
    let outputs = layers |> Array.fold (fun state layer -> layer.runLayer state) data
    // I make the accuracy calculation lazy. This is similar to returning a lambda function that calculates the accuracy
    // although in this case it will be calculated at most once.
    lazy get_accuracy targets.r.P outputs.r.P, squared_error_cost targets outputs 

let train_sgd num_iters learning_rate (layers: FeedforwardLayer[]) =
    [|
    let mutable r' = 0.0f
    let mutable acc = 0.0f
    let base_nodes = layers |> Array.map (fun x -> x.ToArray) |> Array.concat // Stores all the base nodes of the layer so they can later be reset.
    for i=1 to num_iters do
        for x in inputs_outputs do
            let data, target = x
            let lazy_acc,r = training_loop data target layers // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 inputs_outputs.Length) // Adds the cost to the accumulator.
            if System.Single.IsNaN r' then failwith "Nan error"

            for x in base_nodes do x.r.A.setZero() // Resets the base adjoints
            tape.resetTapeAdjoint 0 // Resets the adjoints for the training select
            r.r.A := 1.0f // Pushes 1.0f from the top node
            tape.reversepropTape 0 // Resets the adjoints for the test select
            add_gradients_to_weights' base_nodes learning_rate // The optimization step
            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed for the simple recurrent and feedforward case.

        printfn "The training cost at iteration %i is %f" i r'
        yield r'
        r' <- 0.0f
    |]

train_sgd 10000 0.0005f layers
