// These two decoder function work horribly.
let size_check = 
    [|
    let mutable s = 2UL
    for i=1 to 64 do
        yield s-1UL
        s <- s <<< 1
    |]

let scalar_decoder size x =
    let ar = Array.zeroCreate size
    let mutable x = x |> uint64
    if x > size_check.[size-1] then failwith "x is over what size allows!"
    for i=size-1 downto 0 do
        ar.[i] <- x &&& 1UL |> float32
        x <- x >>> 1
    ar

let array_decoder size (arx: byte[]) =
    let ar = Array.zeroCreate <| size*arx.Length
    for i=0 to arx.Length-1 do
        let mutable x = arx.[i] |> uint64
        if x > size_check.[size-1] then failwith "x is over what size allows!"
        for j = size*(i+1)-1 downto size*i do
            ar.[j] <- x &&& 1UL |> float32
            x <- x >>> 1
    ar

let scalar_encoder (ar: float32[]) =
    let mutable s = 0
    let mutable x = 1
    for i=ar.Length-1 downto 0 do
        s <- s + x*(ar.[i] |> round |> int)
        x <- x <<< 1
    s
