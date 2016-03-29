// Eventually, I will bring the GPU into the picture, but for now let me see if I can compress the 5MB box patterns.

// https://github.com/mrakgr/ML-Dump/blob/master/Spiral%20AD%20Demo%20v0/diffsharp_embedded_reber_v3.fsx

// Last year I did quite a lot of work, so I intend to recycle some of that code.
// I do not need the LSTMs here, but having the utility function for the things I am doing here will come in handy.

#if INTERACTIVE
#r "../packages/DiffSharp.0.7.7/lib/net46/DiffSharp.dll"
#r "../packages/FSharp.Charting.0.90.13/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#endif

open System
open System.Collections.Generic
open DiffSharp.AD.Float32
open DiffSharp.Util

open FSharp.Charting

let rng = Random()

// A layer of neurons
type Layer =
    {
    mutable W:DM  // Input weight matrix
    mutable b:DV  // Bias vector
    a:DM->DM
    } with     // Activation function

    static member private makeUniformRandomDM(hidden_size, input_size) =
        let scale = (2.0f / sqrt(hidden_size+input_size |> float32))
        DV [|for x=1 to hidden_size*input_size do yield (rng.NextDouble()-0.5 |> float32)*scale|]
        |> fun x -> DV.ReshapeToDM(hidden_size,x)

    static member private makeUniformRandomDV hidden_size =
        let scale = (2.0f / sqrt(hidden_size+1 |> float32))
        DV [|for x=1 to hidden_size do yield (rng.NextDouble()-0.5 |> float32)*scale|]

    static member createRandomLayer hidden_size input_size act =
        {
        W = Layer.makeUniformRandomDM(hidden_size, input_size)
        b = Layer.makeUniformRandomDV(hidden_size)

        a = act
        }
     
    member l.ToArray = 
        [|l.W|], [|l.b|]

    member l.tagReverse tag =
         l.W <- l.W |> makeReverse tag
         l.b <- l.b |> makeReverse tag

    member l.addAdjoints (learning_rate: float32) =
         l.W <- l.W.P - learning_rate * l.W.A
         l.b <- l.b.P - learning_rate * l.b.A

    static member fromArray (a : DM[]) act =
        {
         W = a.[0]
         b = a.[1] |> DM.toDV
         a = act
        }

    // For the section with no previous hidden state.
    member l.runLayer (x:DM) =
        l.W*x + l.b |> l.a

let cross_entropy_cost (targets:DM) (inputs:DM) =
    ((targets .* (DM.Log inputs) + (1.0f-targets) .* DM.Log (1.0f-inputs)) |> DM.Sum) / (-inputs.Cols)

let squareSum (targets:DM) (inputs:DM) =
    let r = targets - inputs
    (DM.Pow(r,2) |> DM.Sum) / (2*targets.Cols)

let save_data filename (ar: DM []) =
    use stream_data = IO.File.OpenWrite(filename)
    use writer_data = new IO.BinaryWriter(stream_data)

    // Magic number
    writer_data.Write(929856)

    writer_data.Write(ar.Length)
    for x in ar do
        writer_data.Write(x.Rows)
        writer_data.Write(x.Cols)
        let t = DM.toArray (DM.Transpose x) // Conversion to column major.
        for f in t do writer_data.Write(float32 f)


let load_data file_name =
    let stream_data = IO.File.OpenRead(file_name)
    let reader_data = new IO.BinaryReader(stream_data)

    let m = reader_data.ReadInt32()
    if m <> 929856 then failwith "Wrong file type in load_weights"

    let l = reader_data.ReadInt32()
    let weights = [|
        for i=1 to l do
            let num_rows = reader_data.ReadInt32()
            let num_cols = reader_data.ReadInt32()
            let ar = [| for x=1 to num_cols do yield DV [| for y=1 to num_rows do yield reader_data.ReadSingle() |]|]
            yield DM.ofCols ar // Conversion to row major from column major.
        |]

    reader_data.Close()
    stream_data.Close()
    weights


let _, (inputs, outputs), carr =
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
        (f, IO.File.ReadAllBytes(filename_dat), IO.File.ReadAllBytes(filename_carr) |> Array.map int64)
        |> fun (f, value_function, carr) ->
        f,
        (
        //let histogram = 
        //    Dictionary(HashIdentity.Structural)
        //    |> fun t ->
        //        for x in value_function do
        //            match t.TryGetValue x with
        //            | true, v -> t.[x] <- v+1
        //            | false, _ -> t.[x] <- 1
        //        t |> Seq.toArray |> Array.map (fun x -> x.Key, x.Value)
        //
        //Chart.Bar (histogram, YTitle="Number of Patterns",XTitle="Move Length",Title="Value Function Histogram")
        //|> Chart.Show

        // Yeah, I know this is wrong, but I want to do things in order.
        // Having a single scalar as an input and output will not sit well with the net.
        let outputs =
            value_function
            |> Array.chunkBySize 256
            |> Array.map ( fun x ->
                x 
                |> Array.map float32
                |> DV
                |> DM.ofDV 1)

        let inputs = 
            [|0.0f..value_function.Length-1 |> float32|]
            |> Array.chunkBySize 256
            |> Array.map (fun x ->
                x 
                |> DV
                |> DM.ofDV 1)
                
        inputs, outputs
        ),
        carr


let createNetwork (l:(int * (DM -> DM)) []) =
    [|
    for i=1 to l.Length-1 do
        let l2,a = l.[i]
        let l1,_ = l.[i-1]
        yield Layer.createRandomLayer l2 l1 a
        |] 

let train_feedforward num_iters learning_rate (data: DM[]) (targets: DM[]) (net : Layer[]) =
    let runNetwork (x:DM) (n:Layer[]) =
        Array.fold (fun state (x: Layer) -> x.runLayer state) x n
    [|
    let tag = DiffSharp.Util.GlobalTagger.Next

    let mutable i=1
    let mutable rr=0.0f
    while i <= num_iters && System.Single.IsNaN rr = false do
        let mutable j = 0
        yield [|
            while j < targets.Length && System.Single.IsNaN rr = false do
                for x in net do x.tagReverse tag

                let cost = squareSum targets.[j] (runNetwork data.[j] net)
            
                printfn "The cost is %f at iteration %i, minibatch %i" (float32 cost.P) i j

                rr <- float32 cost.P

                cost |> reverseProp (D 1.0f)

                // Add gradients.
                for x in net do x.addAdjoints learning_rate

                j <- j+1
                yield rr |] |> Array.average
        i <- i+1|]

let net = createNetwork [|1,id;100,tanh;1,sigmoid|]
train_feedforward 1000 5.0f inputs.[0..0] outputs.[0..0] net
//|> Chart.Line
//|> Chart.Show