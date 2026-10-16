# Nexus Communication Protocol

## NCP is a communication protocol that allows Nexus engines to talk with interfaces.

### Commands

#### NCP:
Use the `ncp` command to retrieve information about the engine.

``` text
name Arcturus 1.0
author MC
ncpok
```

#### Go:
Use the `go` command to tell the engine to start thinking.

```text
go [iterations] [temperature] [visits]
```

The optional `visits` argument appends a machine-readable line after the
normal move response:

```text
moveok
Move stone d5 0
visits 0:0 17:0.75 203:0.25
```

Each `action:probability` entry is a legal root action class and its
normalized MCTS visit count. This response is omitted unless `visits` is
requested, preserving the normal command output.

#### Halt:
Use the `halt` command to tell the engine to stop thinking.

#### Stop:
Use the `stop` command to quit the NCP process altogether.

#### Pos:
Resets the position.

#### Move:
Use the `move` command in this format: `Move [MoveType] [coord1] [coord2 if necessary]`

```text
Move stone d5
```
```text
Move stick e4 e5
```

#### Eval:
Use the `eval` command to get an evaluation score of the current position.

#### Random:
Use the `random` command in this format `random <seed>`.
This will

