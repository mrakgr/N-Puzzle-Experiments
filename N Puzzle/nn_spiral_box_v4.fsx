// Yeah, it seems that my mistake of accidentally sizing the minibatches 20 was responsible for this good performance, not unsupervised learning.
// This makes for a decent bit of comedy. Let me try Nesterov's momentum.

// Edit: It makes things go a bit better.

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
        let chunk_size = 2560
        let minibatch_size = 8
        if chunk_size % minibatch_size <> 0 then failwithf "%i %% %i <> 0" chunk_size minibatch_size
        (f, IO.File.ReadAllBytes(filename_dat).[0..chunk_size-1], IO.File.ReadAllBytes(filename_carr) |> Array.map int64)
        |> fun (f, value_function, carr) ->
        f,
        (
        let outputs =
            value_function
            |> Array.chunkBySize minibatch_size
            |> Array.map ( fun x ->
                x 
                |> array_decoder 255
                |> fun x -> DM.makeConstantNode(255,minibatch_size,x)
                )

        let inputs = 
            [|0L..value_function.Length-1 |> int64|]
            |> Array.chunkBySize minibatch_size
            |> Array.map (fun x ->
                x 
                |> Array.map (multinomial_decoder carr >> array_decoder 7)
                |> Array.concat
                |> fun x -> DM.makeConstantNode(7*16,minibatch_size,x)
                )
                
        Array.zip inputs outputs
        ),
        carr

let layers = 
    [|
    FeedforwardLayer.createRandomLayer 1024 112 relu
    FeedforwardLayer.createRandomLayer 1024 1024 relu
    FeedforwardLayer.createRandomLayer 1024 1024 relu
    FeedforwardLayer.createRandomLayer 255 1024 (clipped_steep_sigmoid 3.0f)
    |]

// This does not actually train it, it just initiates the tree for later training.
let training_loop (data: DM) (targets: DM) (layers: FeedforwardLayer[]) =
    let outputs = layers |> Array.fold (fun state layer -> layer.runLayer state) data
    // I make the accuracy calculation lazy. This is similar to returning a lambda function that calculates the accuracy
    // although in this case it will be calculated at most once.
    lazy get_accuracy targets.r.P outputs.r.P, cross_entropy_cost targets outputs 

let train_nag num_iters learning_rate (layers: FeedforwardLayer[]) =
    [|
    let mutable r' = 0.0f
    let mutable acc = 0.0f
    let base_nodes = layers |> Array.map (fun x -> x.ToArray) |> Array.concat // Stores all the base nodes of the layer so they can later be reset.
    let copy_nodes = base_nodes |> Array.map (fun x -> x.r.P.copy())
    let momentum_nodes = copy_nodes |> Array.map (fun x -> dMatrix.create(x.num_rows,x.num_cols) |> fun x -> x.setZero(); x)
    for i=1 to num_iters do
        for x in inputs_outputs do
            let data, target = x
            let lazy_acc,r = training_loop data target layers // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 inputs_outputs.Length) // Adds the cost to the accumulator.
            if System.Single.IsNaN r' then failwith "Nan error"
            acc <- acc + lazy_acc.Value

            for x in base_nodes do x.r.A.setZero() // Resets the base adjoints
            tape.resetTapeAdjoint 0 // Resets the adjoints for the training select
            r.r.A := 1.0f // Pushes 1.0f from the top node
            tape.reversepropTape 0 // Resets the adjoints for the test select
            //add_gradients_to_weights' base_nodes learning_rate // The optimization step
            nesterov_add_gradients base_nodes momentum_nodes copy_nodes learning_rate 0.5f 0.0f
            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed for the simple recurrent and feedforward case.

        printfn "The training cost at iteration %i is %f" i r'
        printfn "The accuracy at iteration is %f" acc
        yield acc
        r' <- 0.0f
        acc <- 0.0f
    |]

train_nag 250 0.005f layers
|> Chart.FastLine
|> Chart.Show
