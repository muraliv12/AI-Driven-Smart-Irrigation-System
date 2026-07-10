# System Architecture

This document describes the end-to-end architecture of the AI-Driven Smart
Irrigation System: how sensor data flows into the RL agent, how the agent
is trained and evaluated, and how the components map onto the physical
Arduino Uno hardware.

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Field["Physical Field Hardware"]
        DHT11["DHT11<br/>Temp + Humidity"]
        LDR["LDR<br/>Light Sensor"]
        SOIL["Capacitive<br/>Soil Moisture"]
        RAIN["Rain Sensor"]
        BATT["Battery Monitor"]
        BARO["Barometric<br/>Pressure"]
        PUMP["Solenoid Valve /<br/>Water Pump"]
        UNO["Arduino Uno"]
    end

    subgraph Ingestion["Data Ingestion Layer"]
        SIM["SensorSimulator<br/>(offline mode)"]
        SER["ArduinoSerialReader<br/>(PySerial, online mode)"]
    end

    subgraph Core["RL Core"]
        ENV["IrrigationEnv<br/>(state fusion + discretization)"]
        REWARD["Reward Module<br/>(Smart Irrigation Score)"]
        AGENT["QLearningAgent<br/>(Q-table)"]
    end

    subgraph Analytics["Analytics Layer"]
        METRICS["Metrics / Baseline<br/>(rule-based comparison)"]
        DASH["Matplotlib Dashboard<br/>(convergence, SIS, water usage)"]
    end

    DHT11 --> UNO
    LDR --> UNO
    SOIL --> UNO
    RAIN --> UNO
    BATT --> UNO
    BARO --> UNO
    UNO -- "Serial CSV @ 9600 baud" --> SER
    SER --> ENV
    SIM --> ENV
    ENV --> REWARD
    REWARD --> AGENT
    AGENT -- "irrigate / hold" --> ENV
    AGENT -- "action" --> UNO
    UNO --> PUMP
    ENV --> METRICS
    METRICS --> DASH
```

## 2. Data Flow

```mermaid
flowchart LR
    A[Raw Sensor Reading] --> B["State Discretization<br/>(5x4x4x3 bins)"]
    B --> C["State Tuple<br/>(moisture, temp, humidity, light)"]
    C --> D{"Epsilon-Greedy<br/>Policy"}
    D -->|explore| E["Random Action"]
    D -->|exploit| F["argmax Q(s,a)"]
    E --> G["Action: Irrigate / Hold"]
    F --> G
    G --> H["Environment Step<br/>(apply action)"]
    H --> I["Next Sensor Reading"]
    I --> J["Smart Irrigation Score (SIS)"]
    J --> K["Symmetric Reward"]
    K --> L["Q-Table Update<br/>(TD Learning)"]
    L --> C
```

## 3. Training Workflow

```mermaid
flowchart TD
    Start(["Start Training"]) --> Init["Initialize Environment + Agent"]
    Init --> EpisodeLoop{"episode <= n_episodes?"}
    EpisodeLoop -->|yes| Reset["env.reset()"]
    Reset --> StepLoop{"step <= max_steps?"}
    StepLoop -->|yes| SelectAction["agent.select_action(state)"]
    SelectAction --> EnvStep["env.step(action)"]
    EnvStep --> ComputeReward["compute reward via SIS"]
    ComputeReward --> UpdateQ["agent.update(s, a, r, s', done)"]
    UpdateQ --> AdvanceStep["state = next_state"]
    AdvanceStep --> StepLoop
    StepLoop -->|episode done| DecayEpsilon["agent.decay_epsilon()"]
    DecayEpsilon --> LogMetrics["log EpisodeMetrics"]
    LogMetrics --> Checkpoint{"episode % checkpoint_every == 0?"}
    Checkpoint -->|yes| SaveQ["agent.save(q_table_file)"]
    Checkpoint -->|no| EpisodeLoop
    SaveQ --> EpisodeLoop
    EpisodeLoop -->|no, done| FinalSave["Final agent.save() + history.to_csv()"]
    FinalSave --> Plots["Generate convergence + SIS plots"]
    Plots --> End(["Training Complete"])
```

## 4. Class Diagram

```mermaid
classDiagram
    class SensorReading {
        +int timestamp_step
        +float soil_moisture
        +float temperature
        +float humidity
        +float light_intensity
        +bool rain_detected
        +float battery_voltage
        +float ambient_pressure
        +to_dict() dict
    }

    class SensorSimulator {
        -Random _rng
        -float _soil_moisture
        +int steps_per_day
        +reset() None
        +read(step, irrigated_last_step) SensorReading
    }

    class ArduinoSerialReader {
        -Serial _conn
        -int _step_counter
        +str port
        +int baud_rate
        +connect() None
        +disconnect() None
        +read(retries) SensorReading
    }

    class IrrigationEnv {
        +int N_ACTIONS
        -SensorSimulator simulator
        -ndarray _moisture_edges
        +reset() State
        +step(action) tuple
        -_discretize(reading) State
    }

    class QLearningAgent {
        +int n_actions
        +float alpha
        +float gamma
        +float epsilon
        -dict q_table
        +select_action(state, greedy) int
        +update(s, a, r, s_next, done) None
        +decay_epsilon() None
        +save(path) None
        +load(path) None
    }

    class TrainingHistory {
        +list episodes
        +add(metrics) None
        +to_csv(path) None
        +rewards() list
        +water_usage() list
    }

    SensorSimulator ..> SensorReading : creates
    ArduinoSerialReader ..> SensorReading : creates
    IrrigationEnv --> SensorSimulator : uses
    IrrigationEnv ..> SensorReading : consumes
    QLearningAgent --> IrrigationEnv : acts on
    TrainingHistory ..> QLearningAgent : records metrics from
```

## 5. Sequence Diagram — Single Training Step

```mermaid
sequenceDiagram
    participant T as train_agent()
    participant E as IrrigationEnv
    participant S as SensorSimulator
    participant R as reward.py
    participant A as QLearningAgent

    T->>E: step(action)
    E->>S: read(step, irrigated_last_step)
    S-->>E: SensorReading
    E->>R: symmetric_reward(reading, action)
    R-->>E: reward (float)
    E->>E: _discretize(reading) -> next_state
    E-->>T: (next_state, reward, done, info)
    T->>A: update(state, action, reward, next_state, done)
    A->>A: Q(s,a) += alpha * (target - Q(s,a))
    A-->>T: (Q-table updated)
```

## 6. Deployment Diagram

```mermaid
graph TB
    subgraph FieldSite["Field Deployment"]
        ArduinoUno["Arduino Uno<br/>+ Sensor Array"]
        SolenoidValve["Solenoid Valve"]
    end

    subgraph EdgeDevice["Edge Device (Raspberry Pi / Laptop)"]
        PySerialProc["Python Process<br/>src.main live"]
        QTable["Trained Q-Table<br/>(models/q_table.pkl)"]
    end

    subgraph DevMachine["Development Machine"]
        TrainingPipeline["Training Pipeline<br/>src.main train"]
        Dashboard["Analytics Dashboard<br/>outputs/*.png"]
    end

    ArduinoUno -- "USB Serial" --> PySerialProc
    PySerialProc -- "loads" --> QTable
    PySerialProc -- "action command" --> ArduinoUno
    ArduinoUno --> SolenoidValve
    TrainingPipeline -- "produces" --> QTable
    TrainingPipeline -- "produces" --> Dashboard
    QTable -. "deployed to" .-> EdgeDevice
```

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `SensorSimulator` | Generate temporally-coherent synthetic sensor data for offline training/demo. |
| `ArduinoSerialReader` | Parse real-time CSV sensor payloads streamed over USB serial from the Arduino Uno. |
| `IrrigationEnv` | Fuse raw sensor values into a discretized state, apply actions, compute rewards, and manage episode termination. |
| `reward.py` | Compute the Smart Irrigation Score (SIS) and the symmetric reward signal. |
| `QLearningAgent` | Maintain the Q-table, select actions via epsilon-greedy policy, and apply the Q-Learning update rule. |
| `metrics.py` | Track per-episode training metrics and run the rule-based baseline for comparison. |
| `dashboard.py` | Render all Matplotlib visualizations used in the README and project report. |
| `main.py` | CLI entrypoint tying together training, evaluation, and live-inference workflows. |
