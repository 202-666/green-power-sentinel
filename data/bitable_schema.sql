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

-- 工单表（新增，D1 风险点：与 BitableClient.work_order_to_fields() 双向一致）
CREATE TABLE work_orders (
    work_order_id TEXT PRIMARY KEY,       -- 工单编号
    alert_id TEXT,                         -- 关联预警ID
    device_id TEXT,                        -- 设备编号
    device_name TEXT,                      -- 设备名称
    fault_type TEXT,                       -- 故障类型
    fault_description TEXT,                -- 故障描述
    recommended_repair_plan TEXT,          -- 推荐维修方案
    required_tools TEXT,                   -- 所需工具
    required_parts TEXT,                   -- 所需备件
    estimated_duration_min FLOAT,          -- 预计耗时（分钟）
    priority TEXT,                         -- 优先级（单选：高/中/低）
    matched_case_id TEXT,                  -- 匹配案例ID
    status TEXT,                           -- 工单状态（单选：待分配/处理中/已完成）
    suggested_deadline DATETIME,           -- 建议截止时间
    created_at DATETIME                    -- 创建时间
);
