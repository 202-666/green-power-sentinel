-- 运行数据表
CREATE TABLE runtime_data (
    record_id TEXT PRIMARY KEY,
    timestamp DATETIME,
    device_id TEXT,
    device_name TEXT,
    device_type TEXT,
    furnace_temperature FLOAT,
    flue_gas_temperature FLOAT,
    steam_pressure FLOAT,
    steam_flow FLOAT,
    bearing_vibration FLOAT,
    bearing_temperature FLOAT,
    so2_concentration FLOAT,
    nox_concentration FLOAT,
    grate_speed FLOAT,
    feed_rate FLOAT,
    oxygen_content FLOAT,
    furnace_pressure FLOAT,
    cooling_water_temp FLOAT,
    batch_no TEXT,
    data_quality_flag TEXT,
    qc_note TEXT
);

-- 预警事件表
CREATE TABLE alert_events (
    alert_id TEXT PRIMARY KEY,
    trigger_time DATETIME,
    device_id TEXT,
    device_name TEXT,
    fault_type TEXT,
    risk_level TEXT,
    confidence FLOAT,
    detection_method TEXT,
    abnormal_params TEXT,
    param_values TEXT,
    predicted_advance_min FLOAT,
    alert_status TEXT,
    push_time DATETIME,
    confirm_time DATETIME,
    handler TEXT,
    work_order_id TEXT,
    feedback TEXT
);

-- 故障知识库表
CREATE TABLE knowledge_base (
    case_id TEXT PRIMARY KEY,
    fault_type TEXT,
    fault_subtype TEXT,
    symptom_pattern TEXT,
    description TEXT,
    cause_analysis TEXT,
    repair_plan TEXT,
    required_tools TEXT,
    required_parts TEXT,
    estimated_duration_min FLOAT,
    severity TEXT,
    historical_frequency TEXT,
    source TEXT,
    last_updated DATETIME,
    reference_count FLOAT
);