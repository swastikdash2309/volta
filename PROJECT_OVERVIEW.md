# VOLTA, in plain language

## The problem

Electric cars are wonderful for the planet, but only if they charge at the right
time. When lots of cars plug in at once, two bad things happen. They strain the
local power grid, the way every air conditioner switching on at once strains it
on a hot day. And they often pull electricity at the dirtiest moments, because at
peak hours the grid burns extra fossil fuel to keep up. So the same car can be
clean or dirty depending purely on when it charges.

## What VOLTA does

VOLTA is software that acts as a smart coordinator for a whole fleet of electric
cars. Instead of every car charging the instant it plugs in, VOLTA decides the
best moment for each car to draw power, so the fleet soaks up clean, cheap energy
from the sun and wind and eases the strain on the grid, while still making sure
every driver has enough charge by the time they need to leave.

The interesting part is that nobody writes the rules for how to do this. The cars
**learn** it themselves, by practicing over thousands of simulated days, the way
you would get better at a game by playing it many times. And they learn to
cooperate, taking turns on the shared grid, even though no central boss tells
them to.

## What we actually built

- A realistic **simulator** of a fleet, a power grid, sunlight, demand, prices,
  and pollution levels, so everything can be tested safely on a computer.
- The **learning brain** that the cars use to decide when to charge. We built two
  versions: a simpler one first, then a far more capable one based on a neural
  network (the same kind of technology behind modern artificial intelligence),
  which we wrote from the ground up.
- A **privacy mode**: in the real world, each parking depot would not want to hand
  over its drivers' private travel history. So we built a way for depots to learn
  together and share only the lessons, never the raw personal data.
- A **forecaster** that predicts how clean the grid will be over the next day.
- A **website** you can open in a browser to watch the fleet coordinate in real
  time, with a live backend that can run any scenario on demand.

## How well it works

Tested over 50 different days it had never seen before, the smart system cut the
fleet's charging pollution by about **41 percent** compared with cars charging
whenever they plugged in, and it also lowered the cost, while still getting nearly
every driver fully charged. The advanced neural-network version clearly beat the
simpler version. These numbers come with proper statistical confidence, not a
single lucky day.

We were also honest about what did not work. The forecaster was accurate, but
giving it to the cars barely helped, because the cars could already read most of
what they needed from the grid's current state. We report that openly rather than
hide it, because knowing when a tool does not help is as valuable as knowing when
it does.

## Does it hold up on real data?

Yes. We checked the idea against real electricity-grid pollution data from the UK
National Grid for four days across the seasons. On those real days, charging at
the cleanest time instead of a random time cut the pollution of charging by about
36 percent, which lines up with what the simulator predicted. Real data also
revealed something a simulation might miss: the cleanest time to charge is not
fixed. It jumped from the middle of the night, to mid-morning, to early evening on
different days, depending on the weather. That is the whole reason a system that
learns and adapts beats a simple timer.

Scaled up to the size of the United States electric bus fleet, a system like this
would avoid roughly fifty thousand tonnes of carbon dioxide a year, about the same
as taking eleven thousand cars off the road. (This is a careful estimate from
public figures, not a deployed pilot, and we are clear about that.)

## The computer science inside it

This one project touches an unusually wide range of serious computer science:

- **Reinforcement learning**: teaching software to make a sequence of decisions by
  trial, error, and reward.
- **Deep learning**: a neural network with its training math (forward pass,
  backpropagation, and a modern optimizer) written from scratch, no shortcuts.
- **Multi-agent systems**: many independent learners sharing one environment and
  having to coordinate.
- **Federated learning**: training a shared model across separate sites without
  ever centralizing private data.
- **Time-series forecasting**: predicting a changing signal into the future and
  measuring the prediction with a proper skill score.
- **Optimization and fairness**: balancing competing goals (clean, cheap, on-time,
  fair) and measuring fairness rigorously.
- **Full-stack software engineering**: a web backend, an interactive front-end,
  and a clean, modular codebase.
- **DevOps and deployment**: containerization and automated publishing so the
  whole thing runs with one command.
- **Software testing and verification**: an automated test suite, a physics and
  correctness audit, and a headless check that the website runs without errors.
- **Statistics**: every result reported with confidence intervals.

## How to run it

```bash
make install      # one-time setup
make all          # trains, builds the dashboard, benchmarks, and verifies
make serve        # then open http://localhost:8000
```

Everything is reproducible from open code. License: MIT.
