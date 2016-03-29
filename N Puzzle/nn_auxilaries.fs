// I need to turn singleton scalar inputs and outputs into something the network can chew.
module NNAuxilaries

open System

let fact =
    [|
    yield 1L

    let mutable s = 1L
    for i=2L to 21L do
        yield s
        s <- s*i
    |]

let multinomial (carr : int64[]) = 
    let u = carr |> Array.sum |> fun x -> fact.[x |> int]
    let d = //carr |> Array.fold (fun s e -> s*fact.[e |> int]) 1L
        let mutable s = fact.[carr.[0] |> int] // Speed optimization
        for i=1 to carr.Length-1 do
            s <- s*fact.[carr.[i] |> int]
        s
    u/d

let multinomial_decoder carr k =
    let num_vars = Array.sum carr
    let result = Array.zeroCreate (num_vars |> int)
    let rec multinomial_decoder (carr : int64[]) k ind =
        if ind < num_vars then
            let m = multinomial carr 
            // Filters out zeroes, scans and appends the index position of the variable in the carr array.
            let mutable s = 0L
            let rec findBack i =
                if i < carr.Length then
                    let t = s
                    s <- s + carr.[i] * m / (num_vars - ind (* The number of vars in the current coefficient array. *))
                    if k < s then t, i else findBack <| i+1
                else s, i
            findBack 0
            |> fun (l,i) ->
                let next_k = k - l
                let next_carr =
                    carr.[i] <- carr.[i]-1L; carr
                result.[ind |> int] <- i |> byte
                multinomial_decoder next_carr next_k (ind+1L)
        else result
    multinomial_decoder (carr |> Array.copy) k 0L

let multinomial_encoder carr (str : byte[]) =
    let num_vars = Array.sum carr // This attempted optimization does not seem to be doing any better than summing the array, but nevermind it.
    let rec multinomial_encoder (carr : int64[]) (str : byte[]) ind lb =
        if ind < num_vars then
            let n = str.[ind |> int] |> int
            let m = multinomial carr 
            let mutable v = 0L
            for i=0 to n-1 do
                v <- v + carr.[i] * m / (num_vars - ind (* The number of vars in the current coefficient array. *) )

            let next_lb = lb+v
            let next_carr =
                carr.[n] <- carr.[n]-1L
                if carr.[n] < 0L then failwith "Invalid string given."
                carr
            multinomial_encoder next_carr str (ind+1L) next_lb
        else lb
    multinomial_encoder (carr |> Array.copy) str 0L 0L

let scalar_decoder size x =
    let ar = Array.zeroCreate size
    if x >= byte size then failwith "x is over what size allows!"
    ar.[int x] <- 1.0f
    ar

let array_decoder size (arx: byte[]) =
    let ar = Array.zeroCreate <| size*arx.Length
    for i=0 to arx.Length-1 do
        let b = i*size
        if int arx.[i] >= size then failwith "x is over what size allows!"
        ar.[b + int arx.[i]] <- 1.0f
    ar


let private test() =
    let stopwatch = Diagnostics.Stopwatch.StartNew()
    let carr = [|10L; 1L; 1L; 1L; 1L; 1L; 1L|]
    let e = multinomial carr |> int
    for i=0 to e-1 do
        let t1 = 
            multinomial_decoder carr 1203L
            |> Array.map (scalar_decoder <| carr.Length+1)
            |> Array.concat

        let t2 =
            multinomial_decoder carr 1203L
            |> (array_decoder <| carr.Length+1)

        if t1 <> t2 then failwith "t1 <> t2"
    printfn "Time it took to verify the two functions are identical: %A" stopwatch.Elapsed
    stopwatch.Restart()


    for i=0 to 1000000 do
        multinomial_decoder carr 1203L
        |> Array.map (scalar_decoder <| carr.Length+1)
        |> Array.concat
        |> Array.map float32
        |> ignore

    printfn "Time it took speed test the first function: %A" stopwatch.Elapsed
    stopwatch.Restart()

    for i=0 to 1000000 do
        multinomial_decoder carr 1203L
        |> (array_decoder <| carr.Length+1)
        |> ignore

    printfn "Time it took speed test the second function: %A" stopwatch.Elapsed

