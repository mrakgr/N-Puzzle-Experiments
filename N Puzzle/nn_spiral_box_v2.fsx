// Tried MXNet, but I cannot get it to work with the GPU which is in line with my other attempts to get DL libraries working (TensorFlow and Diffsharp being the exception).
// Moving on, I'll try pretraining this time. I am not sure how well this will work though, but hopefully it will make training easier.

// I am not sure I am approaching this problem from the right angle.

// The N puzzle is similar to RL problems except here I can calculate the optimal heuristic for every single point whereas in RL I would have to back up
// the reward value with a lambda. Just how did the DeepMind guys managed to get it to work for Go? What do I need to do to get to that level?

(*
Bingo!

It seems that what I was missing was [in fact pretraining](http://imgur.com/a/DrCKR). I do not see it used these days and for good reason - 
on classification tasks the improvements it gives are rather trivial, maybe a 1-2% boost to accuracy at some expense to running time.

Here, on this brutal dataset, the difference unsupervised learning makes is mindblowing - on the 2560 minibatch that I am trying to overfit on it 
gets me from 50% to 90% in less than quarter of the time it takes the purely supervised method to get to 50% with three (2h+1o) layers. This 
stunning result pretty much makes all the months I spent studying autoencoders all worth it. I must have did it all for this. I am not 
disappointed with the neural net at all anymore.

The result I am getting now is what I would expect to get with three hidden layers and I can clearly see the boost in capacity over having 
only an output layer.
*)

// The benefit pretraining gives is the most dramatic I've seen to date. I had no idea this was possible.
// At any rate, this gives me a way forward. At best what I have now is the feedforward net working properly.

// The training is rather slow because the WTA autoencoder cannot handle inputs with large columns.
// It might be worth replacing it with relu + dropout.

// Before that to get the full benefit from sparsity, I need convolutional nets. Even more than the WTA function, they are the
// best sparsity constraint in current time.

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
        let mini_batch_size = 
            let t = 128
            if chunk_size % t = 0 then chunk_size / t else failwithf "%i %% %i <> 0" chunk_size t
        (f, IO.File.ReadAllBytes(filename_dat).[0..chunk_size-1], IO.File.ReadAllBytes(filename_carr) |> Array.map int64)
        |> fun (f, value_function, carr) ->
        f,
        (
        let outputs =
            value_function
            |> Array.chunkBySize mini_batch_size
            |> Array.map ( fun x ->
                x 
                |> array_decoder 255
                |> fun x -> DM.makeConstantNode(255,mini_batch_size,x)
                )

        let inputs = 
            [|0L..value_function.Length-1 |> int64|]
            |> Array.chunkBySize mini_batch_size
            |> Array.map (fun x ->
                x 
                |> Array.map (multinomial_decoder carr >> array_decoder 7)
                |> Array.concat
                |> fun x -> DM.makeConstantNode(7*16,mini_batch_size,x)
                )
                
        Array.zip inputs outputs
        ),
        carr

let l1 = FeedforwardLayer.createRandomLayer 1024 112 (WTA 6)
let l2 = FeedforwardLayer.createRandomLayer 1024 1024 (WTA 6)
let l3 = FeedforwardLayer.createRandomLayer 1024 1024 (WTA 6)
let l4 = InverseFeedforwardLayer.createRandomLayer l3 id // No nonlinearity at the end. Linearities in the final layet cause the individual layers to overfit too badly.
let l5 = InverseFeedforwardLayer.createRandomLayer l2 id
let l6 = InverseFeedforwardLayer.createRandomLayer l1 id

let l1' = FeedforwardLayer.fromArray l1.ToArray relu // Makes supervised layers from the same weights.
let l2' = FeedforwardLayer.fromArray l2.ToArray relu
let l3' = FeedforwardLayer.fromArray l3.ToArray relu
let l_sig = FeedforwardLayer.createRandomLayer 255 1024 (clipped_steep_sigmoid 3.0f)

let layers_deep_autoencoder = [|[|l1;l2;l3|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l4;l5;l6|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The deep autoencoder is not used in this example, but only serves an illustration here.
let layers_1 = [|[|l1|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l6|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The correct functions will get called with dynamic dispatch.
let layers_2 = [|[|l1;l2|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l5|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The correct functions will get called with dynamic dispatch.
let layers_3 = [|[|l1;l2;l3|] |> Array.map (fun x -> x :> IFeedforwardLayer);[|l4|] |> Array.map (fun x -> x :> IFeedforwardLayer);|] |> Array.concat // Upcasting to the base type. The correct functions will get called with dynamic dispatch.
let layers_fine_tune = [|l1';l2';l3';l_sig|] |> Array.map (fun x -> x :> IFeedforwardLayer)

let loop_1 data targets = // These loops are closures. They are not called directly, but passed as parameters into the training function. This one is for the first autoencoder
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data layers_1 // Scan is like fold except it returns the intermediates.
    let inp = outputs.[outputs.Length-3]
    let out = outputs.[outputs.Length-1]
    squared_error_cost inp out, None

let loop_2 data targets = // The targets do nothing in autoencoders, they are here so the type for the supervised net squares out. This one is for the second.
    let l,r = layers_2 |> Array.splitAt 1
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data l // Scan is like fold except it returns the intermediates.
    tape.Add(BlockReverse()) // This blocks the reverse pass from running past this point. It is so the gradients get blocked and only the top two layers get trained.
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) (outputs |> Array.last) r // Scan is like fold except it returns the intermediates.
    let inp = outputs.[outputs.Length-3]
    let out = outputs.[outputs.Length-1]
    squared_error_cost inp out, None

let loop_3 data targets =
    let l,r = layers_3 |> Array.splitAt 2
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data l // Scan is like fold except it returns the intermediates.
    tape.Add(BlockReverse()) // This blocks the reverse pass from running past this point. It is so the gradients get blocked and only the top two layers get trained.
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) (outputs |> Array.last) r // Scan is like fold except it returns the intermediates.
    let inp = outputs.[outputs.Length-3]
    let out = outputs.[outputs.Length-1]
    squared_error_cost inp out, None

let loop_3b data targets = // This is not for the autoencoder, but for the final logistic regression layer. We train it separately first so it does not distrupt the pretrained weights below it.
    let l,r = layers_fine_tune |> Array.splitAt 3
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data l // Scan is like fold except it returns the intermediates.
    tape.Add(BlockReverse()) // This blocks the reverse pass from running past this point. It is so the gradients get blocked and only the top two layers get trained.
    let outputs = Array.scan(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) (outputs |> Array.last) r // Scan is like fold except it returns the intermediates.
    let out = outputs.[outputs.Length-1]
    squared_error_cost targets out, None

let loop_fine_tune data targets = // The full net with the pretrained weights.
    let outputs = Array.fold(fun state (layer:IFeedforwardLayer) -> (layer.runLayer state)) data layers_fine_tune
    cross_entropy_cost targets outputs, Some (lazy get_accuracy targets.r.P outputs.r.P)

// It might be possible to get more speed by not repeating needless calculations in the lower layers, but that would require switching
// branches and some modifying the training loop, so this is decent enough.
// Doing it like this is in fact the most effiecient from a memory standpoint.
let train_sgd num_iters learning_rate training_loop (layers: IFeedforwardLayer[]) =
    [|
    let mutable r' = 0.0f
    let mutable acc = 0.0f
    let base_nodes = layers |> Array.map (fun x -> x.ToArray) |> Array.concat // Stores all the base nodes of the layer so they can later be reset.
    for i=1 to num_iters do
        for x in inputs_outputs do
            let data, target = x
            let (r:Df), lazy_acc = training_loop data target // Builds the tape.

            tape.forwardpropTape 0 // Calculates the forward values. Triggers the ff() closures.
            r' <- r' + (!r.r.P/ float32 inputs_outputs.Length) // Adds the cost to the accumulator.
            if System.Single.IsNaN r' then failwith "Nan error"

            match lazy_acc with
            | Some (lazy_acc: Lazy<floatType>) -> acc <- acc+lazy_acc.Value // Here the accuracy calculation is triggered by accessing it through the Lazy property.
            | None -> ()

            for x in base_nodes do x.r.A.setZero() // Resets the base adjoints
            tape.resetTapeAdjoint 0 // Resets the adjoints for the training select
            r.r.A := 1.0f // Pushes 1.0f from the top node
            tape.reversepropTape 0 // Resets the adjoints for the test select
            add_gradients_to_weights' base_nodes learning_rate // The optimization step
            tape.Clear 0 // Clears the tape without disposing it or the memory buffer. It allows reuse of memory for a 100% gain in speed for the simple recurrent and feedforward case.

        printfn "The training cost at iteration %i is %f" i r'
        if acc <> 0.0f then 
            printfn "The accuracy is %i" (int acc); 
            yield acc
        r' <- 0.0f
        acc <- 0.0f
    |]

let mutable loop_iter = 1

for loop,layers,num_iters,learning_rate in [|loop_1,layers_1,100,0.005f;loop_2,layers_2,100,0.005f;loop_3,layers_3,100,0.005f;loop_3b,layers_fine_tune,100,0.005f;loop_fine_tune,layers_fine_tune,1000,0.005f|] do
    printfn "Starting training loop %i..." loop_iter
    let s = train_sgd num_iters learning_rate loop layers

    if s.Length > 0 then
        s
        |> Chart.Line
        |> Chart.Show

    loop_iter <- loop_iter+1