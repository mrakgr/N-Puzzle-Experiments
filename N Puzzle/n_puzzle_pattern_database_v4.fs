// This version rotates the puzzle 180 degress before solving it in order to solve the other invariant that the Algorithms assignment uses.

open System
open System.Collections.Generic

#if INTERACTIVE
#load "n_puzzle_pattern_builder.fs"
#endif
open PatternBuilder

type Moves =
| UP = 0
| LEFT = 1
| RIGHT = 2
| DOWN = 3

let stopwatch = Diagnostics.Stopwatch.StartNew()
let logging = IO.File.AppendText(IO.Path.Combine(__SOURCE_DIRECTORY__,"logging.log"))
logging.AutoFlush <- true

logging.WriteLine("-----")
logging.WriteLine(sprintf "Starting Session: %A" DateTime.Now)

let value_functions =
    let mapping_function_fringe =
        function
        | 0uy -> 1uy
        | 3uy -> 2uy
        | 7uy -> 3uy
        | 11uy -> 4uy
        | 12uy -> 5uy
        | 13uy -> 6uy
        | 14uy -> 7uy
        | 15uy -> 8uy
        | _ -> 0uy

    let mapping_function_corner =
        function
        | 0uy -> 1uy
        | 8uy -> 2uy
        | 9uy -> 3uy
        | 10uy -> 4uy
        | 12uy -> 5uy
        | 13uy -> 6uy
        | 14uy -> 7uy
        | 15uy -> 8uy
        | _ -> 0uy

    let mapping_function_box_ul =
        function
        | 0uy -> 1uy
        | 1uy -> 2uy
        | 2uy -> 3uy
        | 4uy -> 4uy
        | 5uy -> 5uy
        | 6uy -> 6uy
        | _ -> 0uy

    let mapping_function_box_lr =
        function
        | 0uy -> 1uy
        | 5uy -> 2uy
        | 6uy -> 3uy
        | 8uy -> 4uy
        | 9uy -> 5uy
        | 10uy -> 6uy
        | _ -> 0uy
    [|
    mapping_function_fringe,IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_fringe.dat"), IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_fringe.carr")
    mapping_function_corner,IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_corner.dat"), IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_corner.carr")
    mapping_function_box_ul,IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_box_ul.dat"), IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_box_ul.carr")
    mapping_function_box_lr,IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_box_lr.dat"), IO.Path.Combine(__SOURCE_DIRECTORY__,"mapping_function_box_lr.carr")
    |]
    |> Array.map (
        fun (f,filename_dat,filename_carr) ->
        if IO.File.Exists filename_dat && IO.File.Exists filename_carr 
        then f, (IO.File.ReadAllBytes(filename_dat), IO.File.ReadAllBytes(filename_carr) |> Array.map int64)
        else 
            f, get_value_function_for_pattern 4 f
               |> fun (dat,carr as x) ->
                  IO.File.WriteAllBytes(filename_dat, dat)
                  IO.File.WriteAllBytes(filename_carr, carr |> Array.map byte)
                  x)

logging.WriteLine(sprintf "Time elapsed to calculate (or load) the value functions: %A" stopwatch.Elapsed)
printfn "Time elapsed to calculate (or load) the value functions: %A" stopwatch.Elapsed

let fact =
    [|
    yield 1L

    let mutable s = 1L
    for i=2L to 21L do
        yield s
        s <- s*i
    |]

let run_n_puzzle_solver_on_file file =
    let k, init, init_pos = 
        let t = 
            logging.WriteLine(sprintf "Loading file %s..." file)
            printfn "Loading file %s..." file
            IO.File.ReadAllText(file)
            |> fun x -> x.Split() |> Array.filter ((<>)"") 
        let k = t.[0] |> int

        let size = k*k |> byte

        if k <> 4 then failwith "This solver can only solve 4x4 puzzles."
        if t.Length <> 17 then failwithf "Invalid input in file %s." file

        let init = 
            t.[1..] 
            |> Array.map byte
            |> Array.rev
            |> Array.map (fun x -> (size - x) % size)
        let init_pos =
            init |> Array.findIndex ((=)0uy) // Partial application of the = operator.
                 |> fun x -> x/k,x%k

        k, init, init_pos

    let parity_check =
        let mutable num_inversions = 0
        for i=0 to init.Length-2 do
            let v = init.[i]
            let s = init.[i+1..]
            num_inversions <- 
                s 
                |> Array.map (fun x -> if x < v && x <> 0uy then 1 else 0)
                |> Array.sum
                |> fun x -> x+num_inversions
        if k % 2 = 1 then num_inversions % 2 = 0
        else
            let r = init_pos |> fst
            if r % 2 = 0 then num_inversions % 2 = 0
            else num_inversions % 2 = 1

    if parity_check = false then 
        logging.WriteLine(sprintf "The puzzle %A is unsolvable. Skipping file..." init)
        printfn "The puzzle %A is unsolvable. Skipping file..." init
        0
    else
        let timer = Diagnostics.Stopwatch.StartNew()

        let fringe_search() =
            let mutable max_goal = Int32.MaxValue

            let carr = Array.create (k*k) 1L

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
                            result.[ind |> int] <- i |> int64
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

            let set_trace, is_number_of_steps_less, trace_try_get_value =
                let trace = Dictionary<int64,int>(HashIdentity.Structural)

                let inline set_trace (ar: int64) (i:int) =
                    //let t = multinomial_encoder carr ar
                    match trace.TryGetValue (ar) with
                    | true, v -> if i < v then trace.[ar] <- i
                    | false, _ -> trace.[ar] <- i

                let inline is_number_of_steps_less (ar: int64) c =
                    match trace.TryGetValue (ar) with
                    | true, v -> c < v
                    | false, _ -> true

                let inline trace_try_get_value (ar: int64) =
                    match trace.TryGetValue ar with
                    | true, v -> true, v
                    | false, v -> false, v

                set_trace, is_number_of_steps_less, trace_try_get_value

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

            let victory = multinomial_encoder carr [|0uy..k*k-1 |> byte|]
            let inline check_victory (ar: int64) =
                ar = victory
   
            let inline heuristic_function (ar: _[]) =
                value_functions
                |> Array.map(
                    fun (map_f, (val_f, carr)) ->
                        ar
                        |> Array.map map_f
                        |> multinomial_encoder carr
                        |> fun x -> val_f.[int x]
                        |> fun x -> x)
                |> Array.max
                

            let get_trace_from (p: byte[]) =
                let rec get_trace_from (p: byte[]) (cur_r, cur_c as cur_zero) cur_heuristic accum =
                    let mutable n = None
                    let inline set_least (r, c as swapped_zero) move =
                        if is_viable_swap swapped_zero then
                            let s = swap cur_zero swapped_zero p
                            let s' = s |> multinomial_encoder carr
                            match trace_try_get_value s' with
                            | true, t ->
                                if t <> 0 && t < cur_heuristic then 
                                    match n with
                                    | Some (_,_,t',_) ->
                                        if t < t' then
                                            n <- Some (s, swapped_zero, t, move) // n is assigned in this last expression
                                    | None -> n <- Some (s, swapped_zero, t, move)
                            | false, _ -> ()

                    set_least (-1+cur_r,cur_c) Moves.DOWN // UP
                    set_least (cur_r,-1+cur_c) Moves.RIGHT // LEFT
                    set_least (cur_r,1+cur_c) Moves.LEFT // RIGHT
                    set_least (1+cur_r,cur_c) Moves.UP // DOWN

                    match n with
                    | Some(s, cur , next_heur, move) -> get_trace_from s cur next_heur (move::accum)
                    | None -> accum

                let cur_zero = // position of the blank tile
                    p |> Array.findIndex ((=)0uy) // Partial application of the = operator.
                      |> fun x -> x/k,x%k

                let num_steps_at_p =
                    match trace_try_get_value (multinomial_encoder carr p) with
                    | true, v -> v
                    | false, _ -> failwith "Trace at final state not found!"

                get_trace_from p cur_zero num_steps_at_p []

            let mutable later = Stack(2000)
            let mutable now = Stack(2000)
            let mutable later_upper_bound = Int32.MaxValue
            let mutable final_state = [||]
            let mutable num_ops = 0

            let rec fringe_search (upper_bound : int) =
                if max_goal = Int32.MaxValue then
                    if now.Count > 0 then
                        let (zero_r,zero_c as zero_pos),ar,i,(heuristic_cost: byte) as current_item = now.Pop()
                        let ar' = ar |> multinomial_encoder carr

                        if int heuristic_cost > upper_bound 
                        then 
                            if int heuristic_cost < later_upper_bound then later_upper_bound <- int heuristic_cost
                            later.Push current_item
                            fringe_search upper_bound
                        else

                            let inline if_viable_execute pos =
                                if is_viable_swap pos then 
                                    num_ops <- num_ops+1
                                    let s = swap zero_pos pos ar 
                                    let s' = s |> multinomial_encoder carr
                                    let next_i = i+1
                                    if check_victory s' = false then
                                        if is_number_of_steps_less s' next_i then
                                            let c = heuristic_function s + byte next_i
                                            set_trace s' next_i
                                            if int c <= upper_bound then now.Push((pos,s,next_i,c))
                                            else 
                                                if int c < later_upper_bound then later_upper_bound <- int c
                                                later.Push((pos,s,next_i,c))

                                    else max_goal <- next_i; final_state <- s; set_trace s' next_i
                        

                            if_viable_execute (-1+zero_r,zero_c) // UP
                            if_viable_execute (zero_r,-1+zero_c) // LEFT
                            if_viable_execute (zero_r,1+zero_c) // RIGHT
                            if_viable_execute (1+zero_r,zero_c) // DOWN
                        
                            fringe_search upper_bound
                    else
                        if later.Count = 0 then failwith "later = 0, no path possible!"
                        let t = now
                        now <- later
                        later <- t
                        let t' = later_upper_bound
                        later_upper_bound <- Int32.MaxValue
                        fringe_search t'
            
            let c = heuristic_function init
            now.Push((init_pos,init,1,c))
            let t = multinomial_encoder carr init
            if check_victory t = false then
                set_trace t 1
                fringe_search <| int c

                max_goal-1, get_trace_from final_state
            else
                0, []


        stopwatch.Restart()
        let max_goal, trace = 
            fringe_search()
            |> fun (max_goal, trace as x) -> 
                if max_goal <> trace.Length then failwith "max_goal <> trace.Length"
                x
        logging.WriteLine(sprintf "Time elapsed to solve the N puzzle: %A" stopwatch.Elapsed)
        logging.WriteLine(sprintf "Moves taken: %i" max_goal)
        printfn "Time elapsed to solve the N puzzle: %A" stopwatch.Elapsed
        printfn "Moves taken: %i" max_goal
        1

let stopwatch2 = Diagnostics.Stopwatch.StartNew()

IO.Path.Combine(__SOURCE_DIRECTORY__,"Test Cases for v4")
|> IO.Directory.GetFiles
|> Array.map(
    fun x ->
        run_n_puzzle_solver_on_file x
    )
|> fun x -> 
    logging.WriteLine(sprintf "Solved %i/%i." (x |> Array.sum) (x |> Array.length))
    printfn "Solved %i/%i." (x |> Array.sum) (x |> Array.length)

logging.WriteLine(sprintf "Total time elapsed: %A" stopwatch2.Elapsed)
printfn "Total time elapsed: %A" stopwatch2.Elapsed