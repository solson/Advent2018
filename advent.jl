#!/usr/bin/env julia
push!(LOAD_PATH, @__DIR__)

using Base.Iterators: cycle, filter, flatten, product, Stateful, take
using DataStructures: CircularBuffer, DefaultDict, PriorityQueue, enqueue!, dequeue_pair!, capacity
using Dates
using IterTools: imap, partition, repeatedly
using Match
using OffsetArrays
using StatsBase: counts, span

# TODO(solson): Added to std in Julia 1.1 or 1.2.
eachrow(A) = (view(A, i, :) for i in axes(A, 1))

function main()
  day, part = parse.(Int, ARGS[1:2])
  input_file = get(ARGS, 3, joinpath("input", "day" * string(day, pad = 2)))
  println(advent(Val(day), Val(part), chomp(read(input_file, String))))
end

function advent(::Val{Day}, ::Val{Part}, ::AbstractString) where {Day, Part}
  error("unknown problem: day ", Day, " part ", Part)
end

function advent(::Val{1}, ::Val{1}, input::AbstractString)
  sum(parse.(Int, split(input)))
end

function advent(::Val{1}, ::Val{2}, input::AbstractString)
  diffs = parse.(Int, split(input))
  freq = 0
  seen = Set{Int}(freq)
  for diff in cycle(diffs)
    freq += diff
    freq in seen && return freq
    push!(seen, freq)
  end
end

function advent(::Val{2}, ::Val{1}, input::AbstractString)
  twos, threes = sum(split(input)) do id
    letter_counts = counts(Int.(collect(id)), Int('a'):Int('z'))
    [any(==(2), letter_counts), any(==(3), letter_counts)]
  end
  twos * threes
end

isclose((a, b)) = count(imap(!=, a, b)) == 1

function advent(::Val{2}, ::Val{2}, input::AbstractString)
  ids = split(input)
  a, b = first(filter(isclose, product(ids, ids)))
  String(first.(filter(p -> p[1] == p[2], zip(a, b))))
end

const rx_claim = r"#(\d+) @ (\d+),(\d+): (\d+)x(\d+)"

function advent(::Val{3}, ::Val{1}, input::AbstractString)
  fabric = zeros(Int, 1000, 1000)
  for m in eachmatch(rx_claim, input)
    id, left, top, width, height = parse.(Int, m.captures)
    # Julia uses 1-based indexing.
    cols = range(left + 1, length = width)
    rows = range(top + 1, length = height)
    fabric[cols, rows] .+= 1
  end
  count(x -> x > 1, fabric)
end

function advent(::Val{3}, ::Val{2}, input::AbstractString)
  fabric = fill(-1, 1000, 1000)
  all_ids = Set{Int}()
  hit_ids = Set{Int}()
  for m in eachmatch(rx_claim, input)
    id, left, top, width, height = parse.(Int, m.captures)
    push!(all_ids, id)
    # Julia uses 1-based indexing.
    cols = range(left + 1, length = width)
    rows = range(top + 1, length = height)
    ids = filter(x -> x >= 0, unique(fabric[cols, rows]))
    isempty(ids) || push!(hit_ids, id)
    union!(hit_ids, ids)
    fabric[cols, rows] .= id
  end
  first(setdiff(all_ids, hit_ids))
end

const rx_guard_record = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] (.+)"

function guard_sleeps(input::AbstractString)
  guard_sleep = DefaultDict{Int, OffsetVector{Int}}(() -> zeros(Int, 0:59))
  guard = -1
  prev_time = DateTime(0)

  for line in sort!(split(input, "\n"))
    time_str, msg = match(rx_guard_record, line).captures
    time = DateTime(time_str, dateformat"yyyy-mm-dd HH:MM")
    hour(time) == 23 && (time = ceil(time, Hour))
    if (m = match(r"Guard #(\d+) begins shift", msg)) != nothing
      guard = parse(Int, m[1])
    elseif msg == "wakes up"
      guard_sleep[guard][minute(prev_time) : minute(time) - 1] .+= 1
    end
    prev_time = time
  end

  guard_sleep
end

function advent(::Val{4}, ::Val{1}, input::AbstractString)
  guard_sleep = guard_sleeps(input)
  guard = argmax(Dict(g => sum(mins) for (g, mins) in guard_sleep))
  minute = argmax(guard_sleep[guard])
  guard * minute
end

function advent(::Val{4}, ::Val{2}, input::AbstractString)
  guard_sleep = guard_sleeps(input)
  guard, minute = argmax(Dict(
    ((n, min) = findmax(mins); (g, min) => n) for (g, mins) in guard_sleep
  ))
  guard * minute
end

function parse_polymer(input::AbstractString)
  letter_val = merge(Dict(zip('a':'z', 1:26)), Dict(zip('A':'Z', -(1:26))))
  [letter_val[c] for c in input]
end

function react!(polymer::Vector{Int})
  i = 1
  while i < length(polymer)
    if polymer[i] == -polymer[i + 1]
      deleteat!(polymer, i:i+1)
      i = max(1, i - 1)
    else
      i += 1
    end
  end
  polymer
end

function advent(::Val{5}, ::Val{1}, input::AbstractString)
  length(react!(parse_polymer(input)))
end

function advent(::Val{5}, ::Val{2}, input::AbstractString)
  polymer = parse_polymer(input)
  minimum(length(react!(filter(x -> abs(x) != abs(y), polymer))) for y in 1:26)
end

struct Point
  x::Int
  y::Int
end

Base.Broadcast.broadcastable(p::Point) = Ref(p)
Base.parse(::Type{Point}, s::AbstractString) =
  Point(parse.(Int, split(s, ","))...)
dist(a::Point, b::Point) = abs(a.x - b.x) + abs(a.y - b.y)

function advent(::Val{6}, ::Val{1}, input::AbstractString)
  points = parse.(Point, split(input, "\n"))
  left, right = extrema(p.x for p in points)
  top, bottom = extrema(p.y for p in points)
  isboundary(p::Point) = p.x in [left, right] || p.y in [top, bottom]
  areas = zeros(length(points))

  for x = left:right, y = top:bottom
    p = Point(x, y)
    dists = dist.(p, points)
    d, closest = findmin(dists)
    if count(==(d), dists) == 1
      if isboundary(p)
        areas[closest] = Inf
      else
        areas[closest] += 1
      end
    end
  end

  Int(maximum(filter(isfinite, areas)))
end

function advent(::Val{6}, ::Val{2}, input::AbstractString)
  points = parse.(Point, split(input, "\n"))
  left, right = extrema(p.x for p in points)
  top, bottom = extrema(p.y for p in points)
  count(p -> sum(dist.(Point(p...), points)) < 10000, product(left:right, top:bottom))
end

const rx_prereq = r"Step (\w) must be finished before step (\w) can begin\."

function parse_prereqs(input::AbstractString)
  prereqs = DefaultDict{String, Set{String}}(Set{String})
  for m in eachmatch(rx_prereq, input)
    before, after = m.captures
    push!(prereqs[after], before)
    prereqs[before] # Ensure the other step is at least default initialized.
  end
  prereqs
end

function advent(::Val{7}, ::Val{1}, input::AbstractString)
  prereqs = parse_prereqs(input)
  steps = IOBuffer()
  while !isempty(prereqs)
    step = minimum(keys(filter(p -> isempty(p.second), prereqs)))
    write(steps, step)
    delete!(prereqs, step)
    delete!.(values(prereqs), step)
  end
  String(take!(steps))
end

function advent(::Val{7}, ::Val{2}, input::AbstractString)
  prereqs = parse_prereqs(input)
  queue = PriorityQueue{String, Int}()
  time = 0

  while !isempty(prereqs) || !isempty(queue)
    available = sort!(collect(keys(filter(p -> isempty(p.second), prereqs))))

    # Assign as many free workers to available tasks as possible.
    while length(queue) < 5 && !isempty(available)
      task = popfirst!(available)
      enqueue!(queue, task, time + 60 + (first(task) - 'A' + 1))
      delete!(prereqs, task)
    end

    # Step forward in time and finish a task.
    task, time = dequeue_pair!(queue)
    delete!.(values(prereqs), task)
  end

  time
end

struct Tree
  children::Vector{Tree}
  metadata::Vector{Int}
end

sum_metadata(t::Tree)::Int =
  sum(t.metadata) + (isempty(t.children) ? 0 : sum(sum_metadata, t.children))

function value(t::Tree)
  if isempty(t.children)
    sum(t.metadata)
  else
    sum(value.(t.children[filter(in(keys(t.children)), t.metadata)]))
  end
end

function parse_tree(data::Stateful{Vector{Int}})
  num_children = popfirst!(data)
  num_metadata = popfirst!(data)
  Tree(
    collect(repeatedly(() -> parse_tree(data), num_children)),
    collect(take(data, num_metadata)),
  )
end

function Base.parse(Tree, s::AbstractString)
  parse_tree(Stateful(parse.(Int, split(s))))
end

function advent(::Val{8}, ::Val{1}, input::AbstractString)
  sum_metadata(parse(Tree, input))
end

function advent(::Val{8}, ::Val{2}, input::AbstractString)
  value(parse(Tree, input))
end

function advent(::Val{9}, ::Val{1}, input::AbstractString)
  m = match(r"(\d+) players; last marble is worth (\d+) points", input)
  num_players, last_marble = parse.(Int, m.captures)
  scores = zeros(Int, num_players)
  marbles = [0]
  i = 0

  for (next_marble, player) in zip(1:last_marble, cycle(keys(scores)))
    if next_marble % 23 == 0
      i = mod(i - 7, length(marbles))
      scores[player] += next_marble + splice!(marbles, i + 1)
    else
      i = mod(i + 2, length(marbles))
      insert!(marbles, i + 1, next_marble)
    end
  end

  maximum(scores)
end

mutable struct Marble
  prev::Union{Nothing, Marble}
  next::Union{Nothing, Marble}
  number::Int
end

function advent(::Val{9}, ::Val{2}, input::AbstractString)
  m = match(r"(\d+) players; last marble is worth (\d+) points", input)
  num_players, last_marble = parse.(Int, m.captures)
  scores = zeros(Int, num_players)
  marble = Marble(nothing, nothing, 0)
  marble.prev = marble
  marble.next = marble

  for (next_marble, player) in zip(1:last_marble*100, cycle(keys(scores)))
    if next_marble % 23 == 0
      marble = marble.prev.prev.prev.prev.prev.prev.prev # 7 previous
      scores[player] += next_marble + marble.number
      marble.prev.next = marble.next
      marble.next.prev = marble.prev
      marble = marble.next
    else
      marble = marble.next.next
      new_marble = Marble(marble.prev, marble, next_marble)
      marble.prev.next = new_marble
      marble.prev = new_marble
      marble = new_marble
    end
  end

  maximum(scores)
end

const rx_point = r"
  position=< \s* (-?\d+), \s* (-?\d+) > \s*
  velocity=< \s* (-?\d+), \s* (-?\d+) >
"x

function day10(input::AbstractString)
  points = vcat(map(m -> parse.(Int, m.captures)', eachmatch(rx_point, input))...)
  height = length(span(points[:, 2]))
  seconds = 0

  while true
    # TODO(solson): Use DataFrames for named columns.
    points[:, 1:2] += points[:, 3:4]
    new_height = length(span(points[:, 2]))
    new_height > height && break
    height = new_height
    seconds += 1
  end

  # Undo the last step.
  points[:, 1:2] -= points[:, 3:4]

  # TODO(solson): Simplify!
  width = length(span(points[:, 1]))
  min_x = minimum(points[:, 1])
  min_y = minimum(points[:, 2])
  grid = fill('.', width, height)
  for (x, y) in eachrow(points)
    grid[x - min_x + 1, y - min_y + 1] = '#'
  end
  (grid = join(mapslices(join, grid, dims = 1), "\n"), seconds = seconds)
end

function advent(::Val{10}, ::Val{1}, input::AbstractString)
  day10(input).grid
end

function advent(::Val{10}, ::Val{2}, input::AbstractString)
  day10(input).seconds
end

power_level(serial, x, y) = div(((x + 10) * y + serial) * (x + 10), 100) % 10 - 5

function advent(::Val{11}, ::Val{1}, input::AbstractString)
  serial = parse(Int, input)
  grid = [power_level(serial, x, y) for x = 1:300, y = 1:300]
  x, y = Tuple(argmax([sum(grid[x:x+2, y:y+2]) for x = 1:298, y = 1:298]))
  "$x,$y"
end

function advent(::Val{11}, ::Val{2}, input::AbstractString)
  serial = parse(Int, input)
  grid = zeros(Int, 300, 300, 300)
  grid[1, :, :] = [power_level(serial, x, y) for x = 1:300, y = 1:300]
  for size = 2:300
    @show size
    gap = size - 1
    for x = 1:(300 - gap), y = 1:(300 - gap)
      grid[size, x, y] =
        grid[size - 1, x, y] +
        sum(grid[1, x + gap, range(y, length = gap)]) +
        sum(grid[1, range(x, length = gap), y + gap]) +
        grid[1, x + gap, y + gap]
    end
  end
  size, x, y = Tuple(argmax(grid))
  "$x,$y,$size"
end

function parse_pots(input::AbstractString)
  initial_str, rules_str = split(input, "\n\n")
  pots = collect(split(initial_str, ": ")[2]) .== '#'
  rules = Set{BitVector}()
  for (pattern, result) in split.(split(rules_str, "\n"), " => ")
    result == "#" && push!(rules, collect(pattern) .== '#')
  end
  pots, rules
end

function step_pots(pots::BitArray, rules::Set{BitVector})::BitVector
  patterns = BitArray.(partition(flatten([falses(4), pots, falses(4)]), 5, 1))
  BitArray(map(in(rules), patterns))
end

sum_pots(pots, offset) = sum(findall(pots) .+ offset)
render_pots(pots) = join(map(b -> b ? '#' : '.', pots))

function advent(::Val{12}, ::Val{1}, input::AbstractString)
  pots, rules = parse_pots(input)
  offset = -1
  for _ in 1:20
    offset -= 2
    pots = step_pots(pots, rules)
  end
  sum_pots(pots, offset)
end

function advent(::Val{12}, ::Val{2}, input::AbstractString)
  pots, rules = parse_pots(input)
  recent_sums = CircularBuffer{Int}(5)
  offset = -1
  for step in 1:50_000_000_000
    offset -= 2
    pots = step_pots(pots, rules)
    push!(recent_sums, sum_pots(pots, offset))
    diffs = imap(-, recent_sums[2:end], recent_sums[1:end-1])
    if length(recent_sums) == capacity(recent_sums) && length(unique(diffs)) == 1
      diff = first(diffs)
      return recent_sums[end] + (50_000_000_000 - step) * diff
    end
  end
  sum_pots(pots, offset)
end

@enum Turn left right straight

mutable struct Cart
  pos::NTuple{2, Int}
  vel::NTuple{2, Int}
  next_turn::Turn
end

function advent(::Val{13}, ::Val{1}, input::AbstractString)
  lines = split(input, "\n")
  height = length(lines)
  width = maximum(length.(lines))
  tracks = permutedims(hcat(collect.(rpad.(lines, width))...))
  carts = Cart[]
  cart_types = [('<', (0, -1), '-'), ('>', (0, 1), '-'), ('^', (-1, 0), '|'), ('v', (1, 0), '|')]
  for (c, vel, replacement) in cart_types
    positions = findall(==(c), tracks)
    tracks[positions] .= replacement
    append!(carts, map(pos -> Cart(Tuple(pos), vel, left), positions))
  end

  while true
    sort!(carts, by = c -> c.pos)
    for cart in carts
      new_pos = cart.pos .+ cart.vel
      any(c -> c.pos == new_pos, carts) && return reverse(new_pos .- (1, 1))
      cart.pos = new_pos
      @match tracks[cart.pos...] begin
        '\\' => (cart.vel = reverse(cart.vel))
        '/' => (cart.vel = .-reverse(cart.vel))
        '+' => begin
          if cart.next_turn == left
            cart.next_turn = straight
            (x, y) = cart.vel
            cart.vel = (-y, x)
          elseif cart.next_turn == straight
            cart.next_turn = right
          elseif cart.next_turn == right
            cart.next_turn = left
            (x, y) = cart.vel
            cart.vel = (y, -x)
          end
        end
      end
    end
    # t = copy(tracks)
    # t[[CartesianIndex(c.pos) for c in carts]] .= '#'
    # println(join(join.(eachrow(t)), "\n"))
  end
end

function advent(::Val{13}, ::Val{2}, input::AbstractString)
  lines = split(input, "\n")
  height = length(lines)
  width = maximum(length.(lines))
  tracks = permutedims(hcat(collect.(rpad.(lines, width))...))
  carts = Cart[]
  cart_types = [('<', (0, -1), '-'), ('>', (0, 1), '-'), ('^', (-1, 0), '|'), ('v', (1, 0), '|')]
  for (c, vel, replacement) in cart_types
    positions = findall(==(c), tracks)
    tracks[positions] .= replacement
    append!(carts, map(pos -> Cart(Tuple(pos), vel, left), positions))
  end

  while true
    # t = copy(tracks)
    # t[[CartesianIndex(c.pos) for c in carts if c.vel == (0, -1)]] .= '<'
    # t[[CartesianIndex(c.pos) for c in carts if c.vel == (0, 1)]] .= '>'
    # t[[CartesianIndex(c.pos) for c in carts if c.vel == (-1, 0)]] .= '^'
    # t[[CartesianIndex(c.pos) for c in carts if c.vel == (1, 0)]] .= 'v'
    # println(join(join.(eachrow(t)), "\n"))
    sort!(carts, by = c -> c.pos)
    deleted = Int[]
    for (i, cart) in enumerate(carts)
      i in deleted && continue
      new_pos = cart.pos .+ cart.vel
      collisions = collect(filter(x -> !(x in deleted), findall(c -> c.pos == new_pos, carts)))
      if !isempty(collisions)
        push!(deleted, i)
        append!(deleted, collisions)
      end
      cart.pos = new_pos
      @match tracks[cart.pos...] begin
        '\\' => (cart.vel = reverse(cart.vel))
        '/' => (cart.vel = .-reverse(cart.vel))
        '+' => begin
          if cart.next_turn == left
            cart.next_turn = straight
            (x, y) = cart.vel
            cart.vel = (-y, x)
          elseif cart.next_turn == straight
            cart.next_turn = right
          elseif cart.next_turn == right
            cart.next_turn = left
            (x, y) = cart.vel
            cart.vel = (y, -x)
          end
        end
      end
    end
    unique!(sort!(deleted))
    deleteat!(carts, deleted)
    length(carts) == 1 && return reverse(first(carts).pos .- (1, 1))
  end
end

function advent(::Val{14}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{14}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{15}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{15}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{16}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{16}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{17}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{17}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{18}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{18}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{19}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{19}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{20}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{20}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{21}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{21}, ::Val{2}, input::AbstractString)
  error("todo")
end

function erosion_levels(depth::Int, target_x::Int, target_y::Int, buffer::Int = 0)
  M = 20183
  erosion(geologic::Int)::Int = (geologic + depth) % M
  erosion_levels = zeros(Int, 0:target_x + buffer, 0:target_y + buffer)
  erosion_levels[0, 0] = erosion(0)
  erosion_levels[target_x, target_y] = erosion(0)
  erosion_levels[1:target_x + buffer, 0] = erosion.((1:target_x + buffer) * 16807)
  erosion_levels[0, 1:target_y + buffer] = erosion.((1:target_y + buffer) * 48271)
  for x = 1:target_x + buffer, y = 1:target_y + buffer
    x == target_x && y == target_y && continue
    erosion_levels[x, y] = erosion.(erosion_levels[x - 1, y] * erosion_levels[x, y - 1])
  end
  erosion_levels
end

function advent(::Val{22}, ::Val{1}, input::AbstractString)
  depth, target_x, target_y =
    parse.(Int, match(r"depth: (\d+)\ntarget: (\d+),(\d+)", input).captures)
  sum(erosion_levels(depth, target_x, target_y) .% 3)
end

@enum CaveRegion rocky=0 wet=1 narrow=2

@enum CaveTool none torch climbing_gear

struct CaveState
  pos::CartesianIndex{2}
  tool::CaveTool
end

dist(i::CartesianIndex{2}, j::CartesianIndex{2}) = sum(abs.(Tuple(i - j)))

function advent(::Val{22}, ::Val{2}, input::AbstractString)
  depth, target_x, target_y =
    parse.(Int, match(r"depth: (\d+)\ntarget: (\d+),(\d+)", input).captures)
  cave = CaveRegion.(erosion_levels(depth, target_x, target_y, 200) .% 3)
  directions = CartesianIndex.([(-1, 0), (1, 0), (0, -1), (0, 1)])
  valid_tools = Dict{CaveRegion, Vector{CaveTool}}(
    rocky => [torch, climbing_gear],
    wet => [none, climbing_gear],
    narrow => [none, torch],
  )
  goal = CartesianIndex(target_x, target_y)
  seen = Set{CartesianIndex{2}}()
  origin = CartesianIndex(0, 0)
  fringe = PriorityQueue{CaveState, NTuple{2, Int}}(
    CaveState(origin, torch) => (dist(origin, goal), 0)
  )

  while true
    current, (_, current_time) = dequeue_pair!(fringe)
    push!(seen, current.pos)

    if current.pos == goal
      fringe_positions = map(s -> s.pos, collect(keys(fringe)))
      for (y, row) in enumerate(eachrow(permutedims(cave)))
        for (x, region) in enumerate(row)
          pos = CartesianIndex(x - 1, y - 1)
          c = ['.', '=', '|'][Int(region) + 1]
          pos == goal && (c = 'X')
          if pos == current.pos
            printstyled(IOContext(stdout, :color => true), c, color = :green, bold = true)
          elseif pos in fringe_positions
            printstyled(IOContext(stdout, :color => true), c, color = :green)
          elseif pos in seen
            printstyled(IOContext(stdout, :color => true), c, color = :red)
          else
            print(c)
          end
        end
        println("")
      end
      println("")

      return current.tool == torch ? current_time : current_time + 7
    end

    for d in directions
      pos = current.pos + d
      (pos[1] < 0 || pos[2] < 0) && continue
      current_region = cave[current.pos]
      region = cave[pos]
      if region == current_region
        tool = current.tool
        time_cost = 1
      else
        tool = first(intersect(valid_tools[current_region], valid_tools[region]))
        time_cost = tool == current.tool ? 1 : 8
      end
      state = CaveState(pos, tool)
      time = current_time + time_cost
      if !(pos in seen) && (!(state in keys(fringe)) || fringe[state][2] > time)
        fringe[state] = (time + dist(pos, goal), time)
      end
    end
  end
end

function advent(::Val{23}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{23}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{24}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{24}, ::Val{2}, input::AbstractString)
  error("todo")
end

function advent(::Val{25}, ::Val{1}, input::AbstractString)
  error("todo")
end

function advent(::Val{25}, ::Val{2}, input::AbstractString)
  error("todo")
end

main()
