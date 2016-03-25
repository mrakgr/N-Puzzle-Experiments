// Generates random puzzles.
// Merely permuting an array [|0..k*k-1|] does not work as more than half the instances are unsolvable.

open System

let rng = Random() // Don't put this inside the make_random function.

let make_random k num_random_moves = 
    let inline is_viable_swap (r,c) =
        r >= 0 && c >= 0 && r < k && c < k

    let inline get (ar: byte[]) (r,c as p) = // I could have used 2D arrays, but checking victory conditions would have been such a pain then.
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
    let mutable init = [|0uy..k*k-1 |> byte|]
    let mutable init_pos =
        init |> Array.findIndex ((=)0uy) // Partial application of the = operator.
                |> fun x -> x/k,x%k

    for i=1 to num_random_moves do
        let r,c = init_pos
        let moves = [|-1+r,c; // UP
        r,-1+c; // LEFT
        r,1+c; // RIGHT
        1+r,c|] // DOWN
        let move = moves.[rng.Next(0,4)]
        if is_viable_swap move then
            init <- swap init_pos move init
            init_pos <- move
    k, init_pos, init

let parity_check (k, init_pos, (init : byte[])) =
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

let p = IO.Path.Combine(__SOURCE_DIRECTORY__,"Test Cases")
for i=0 to 99 do
    let k, _, init as x = make_random 4 10000
    x
    |> parity_check // This is to check the parity checker.
    |> fun x -> if x = false then failwith "Parity check failed!"

    let sb = Text.StringBuilder(100)
    sb.Append(sprintf "%i\n" k) |> ignore
    for i=0 to k-1 do
        for j=0 to k-1 do
            sb.Append(sprintf "%i " init.[i*k+j]) |> ignore
        sb.Append(sprintf "\n") |> ignore

    IO.File.WriteAllText(IO.Path.Combine(p,sprintf "puzzle_4x4_%2i.txt" i), sb.ToString())
