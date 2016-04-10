// Returns the value function for the 3x3 puzzle.

module PatternBuilder3x3

#if INTERACTIVE
#load "nn_auxilaries.fs"
#endif
open NNAuxilaries

open System
open System.Collections.Generic

let k = 3
let mapping_function = id
let minibatch_size = 128

let value_function, carr, number_of_examples =
    let final_state, init_pos = 
        [|0uy..k*k-1 |> byte|] // I am working backwards from the final state here.
        |> Array.map mapping_function
        |> fun ar ->
            ar,
            ar  |> Array.findIndex ((=)1uy)
                |> fun x -> x/k,x%k

    let carr = 
        Array.zeroCreate (final_state |> Array.max |> fun x -> x+1uy |> int)
        |> fun carr ->
            final_state |> Array.iter (fun x -> let x = int x in carr.[x] <- carr.[x]+1L)
            carr

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

    // TODO: It might be possible to modify this function to return the (row, column) of a variable. That way I would not need to waste memory
    // passing them in the queue.
    let multinomial_decoder k =
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

    let multinomial_encoder (str : byte[]) =
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

    let bfs() =
        let number_of_values = multinomial carr |> fun x -> if x > (Int32.MaxValue |> int64) then failwith "Too many values!" else x |> int
        let value_function = Array.create number_of_values 255uy//Dictionary(number_of_values,HashIdentity.Structural) // The value function represented by a dictionary.

        let queue = Queue(min number_of_values 100000)
        queue.Enqueue(multinomial_encoder final_state |> int, 0)
        value_function.[multinomial_encoder final_state |> int] <- 0uy
        let mutable values_calculated = 1

        let inline is_viable_swap (r,c) =
            r >= 0 && c >= 0 && r < k && c < k

        let inline get (ar: byte[]) (r,c as p) =
            if is_viable_swap p then ar.[r*k+c]
            else failwith "Invalid array get."

        let inline set (ar: byte[]) (r,c as p) v =
            if is_viable_swap p then ar.[r*k+c] <- v
            else failwith "Invalid array set."

        let inline swap (r1,c1) (r2,c2) (ar: byte[]) =
            let ar = ar.Clone() :?> byte[]
            set ar (r1,c1) (get ar (r1,c1) + get ar (r2,c2))
            set ar (r2,c2) (get ar (r1,c1) - get ar (r2,c2))
            set ar (r1,c1) (get ar (r1,c1) - get ar (r2,c2))
            ar

        let inline is_number_of_steps_less (ar: int64) c =
            match value_function.[int ar] with
            | 255uy -> true
            | v -> c < v
            

        while queue.Count > 0 do // Do BFS from the final state as long as all the values haven't been calculated.
            let (ar': int), i = queue.Dequeue()
            let ar = multinomial_decoder <| int64 ar'
            let one_r, one_c as one_pos =
                ar |> Array.findIndex ((=)1uy) // Partial application of the = operator.
                   |> fun x -> x/k,x%k

            let inline if_viable_execute pos =
                if is_viable_swap pos then 
                    let s = swap one_pos pos ar 
                    let s' = s |> multinomial_encoder
                    let next_i = i+1 

                    if is_number_of_steps_less s' <| byte next_i then
                        values_calculated <- values_calculated+1

                        if values_calculated % 1000000 = 0 then
                            printfn "values_calculated=%i number_of_values=%i thread=%i" values_calculated number_of_values Threading.Thread.CurrentThread.ManagedThreadId

                        value_function.[int s'] <- byte next_i
                        
                        queue.Enqueue(int s',next_i)

            if_viable_execute (-1+one_r,one_c) // UP
            if_viable_execute (one_r,-1+one_c) // LEFT
            if_viable_execute (one_r,1+one_c) // RIGHT
            if_viable_execute (1+one_r,one_c) // DOWN
        value_function

    (bfs(), carr)
    |> fun (value_function, carr) ->
        let adjusted_value_function =
            [|
            for i=0 to value_function.Length-1 do
                if value_function.[i] <> 255uy then yield int64 i |> (multinomial_decoder >> array_decoder 9), value_function.[i] |> scalar_decoder 32
            |]
            |> Array.chunkBySize minibatch_size
            |> Array.map (
                fun x ->
                let l,r = Array.unzip x
                Array.concat l, Array.concat r
                )
        adjusted_value_function, carr, value_function.Length/2

