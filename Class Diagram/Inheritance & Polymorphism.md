<svg viewBox="0 0 900 700" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="900" height="700" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="450" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Inheritance & Exception Hierarchy</text>
  
  <!-- Base Exception (Python built-in) -->
  <rect x="350" y="60" width="200" height="80" fill="#ffeaa7" stroke="#fdcb6e" stroke-width="2" rx="5"/>
  <text x="450" y="85" text-anchor="middle" font-weight="bold" fill="#2c3e50">Exception</text>
  <text x="450" y="105" text-anchor="middle" font-size="12" fill="#636e72">(Python Built-in)</text>
  
  <!-- DatabaseException (Base) -->
  <rect x="350" y="180" width="200" height="100" fill="#fab1a0" stroke="#e17055" stroke-width="2" rx="5"/>
  <text x="450" y="205" text-anchor="middle" font-weight="bold" fill="#2c3e50">DatabaseException</text>
  <line x1="370" y1="215" x2="530" y2="215" stroke="#e17055"/>
  <text x="370" y="235" font-size="11" fill="#2c3e50">+ message: str</text>
  <text x="370" y="250" font-size="11" fill="#2c3e50">+ error_code: str</text>
  <text x="370" y="265" font-size="11" fill="#2c3e50">+ original_exception</text>
  
  <!-- Specific Exceptions -->
  <g>
    <!-- ConnectionException -->
    <rect x="50" y="350" width="150" height="60" fill="#dda0dd" stroke="#9b59b6" stroke-width="2" rx="5"/>
    <text x="125" y="375" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Connection</text>
    <text x="125" y="390" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Exception</text>
    
    <!-- AuthenticationException -->
    <rect x="220" y="350" width="150" height="60" fill="#dda0dd" stroke="#9b59b6" stroke-width="2" rx="5"/>
    <text x="295" y="375" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Authentication</text>
    <text x="295" y="390" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Exception</text>
    
    <!-- TableNotFoundException -->
    <rect x="390" y="350" width="150" height="60" fill="#dda0dd" stroke="#9b59b6" stroke-width="2" rx="5"/>
    <text x="465" y="375" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">TableNotFound</text>
    <text x="465" y="390" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Exception</text>
    
    <!-- TransactionException -->
    <rect x="560" y="350" width="150" height="60" fill="#dda0dd" stroke="#9b59b6" stroke-width="2" rx="5"/>
    <text x="635" y="375" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Transaction</text>
    <text x="635" y="390" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Exception</text>
    
    <!-- More exceptions -->
    <rect x="730" y="350" width="150" height="60" fill="#dda0dd" stroke="#9b59b6" stroke-width="2" rx="5"/>
    <text x="805" y="375" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Timeout</text>
    <text x="805" y="390" text-anchor="middle" font-weight="bold" font-size="12" fill="#2c3e50">Exception</text>
  </g>
  
  <!-- Inheritance arrows -->
  <defs>
    <marker id="inheritance" markerWidth="12" markerHeight="12" refX="12" refY="6" orient="auto">
      <polygon points="0,0 0,12 12,6" fill="white" stroke="#2c3e50" stroke-width="2"/>
    </marker>
  </defs>
  
  <!-- Exception -> DatabaseException -->
  <line x1="450" y1="140" x2="450" y2="180" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
  
  <!-- DatabaseException to specific exceptions -->
  <line x1="400" y1="280" x2="125" y2="350" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
  <line x1="420" y1="280" x2="295" y2="350" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
  <line x1="450" y1="280" x2="465" y2="350" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
  <line x1="480" y1="280" x2="635" y2="350" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
  <line x1="500" y1="280" x2="805" y2="350" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
  
  <!-- Factory Pattern Example -->
  <g transform="translate(50, 470)">
    <text x="0" y="0" font-size="16" font-weight="bold" fill="#2c3e50">Abstract Factory Pattern:</text>
    
    <!-- AbstractDatabaseFactory -->
    <rect x="0" y="20" width="180" height="80" fill="#e8f4fd" stroke="#3498db" stroke-width="2" rx="5"/>
    <text x="90" y="40" text-anchor="middle" font-weight="bold" fill="#2c3e50">«abstract»</text>
    <text x="90" y="55" text-anchor="middle" font-weight="bold" fill="#2c3e50">DatabaseFactory</text>
    <line x1="20" y1="65" x2="160" y2="65" stroke="#3498db"/>
    <text x="20" y="80" font-size="10" fill="#2c3e50">+ create_reader()</text>
    <text x="20" y="92" font-size="10" fill="#2c3e50">+ create_writer()</text>
    
    <!-- OracleFactory -->
    <rect x="220" y="20" width="120" height="80" fill="#e8f8e8" stroke="#27ae60" stroke-width="2" rx="5"/>
    <text x="280" y="45" text-anchor="middle" font-weight="bold" fill="#2c3e50">OracleFactory</text>
    <line x1="240" y1="55" x2="320" y2="55" stroke="#27ae60"/>
    <text x="240" y="70" font-size="10" fill="#2c3e50">Concrete impl.</text>
    <text x="240" y="82" font-size="10" fill="#2c3e50">for Oracle</text>
    
    <!-- SnowflakeFactory -->
    <rect x="360" y="20" width="120" height="80" fill="#e8f8e8" stroke="#27ae60" stroke-width="2" rx="5"/>
    <text x="420" y="45" text-anchor="middle" font-weight="bold" fill="#2c3e50">SnowflakeFactory</text>
    <line x1="380" y1="55" x2="460" y2="55" stroke="#27ae60"/>
    <text x="380" y="70" font-size="10" fill="#2c3e50">Concrete impl.</text>
    <text x="380" y="82" font-size="10" fill="#2c3e50">for Snowflake</text>
    
    <!-- Inheritance arrows for factories -->
    <line x1="180" y1="60" x2="220" y2="60" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
    <line x1="180" y1="60" x2="360" y2="60" stroke="#2c3e50" stroke-width="2" marker-end="url(#inheritance)"/>
  </g>
  
  <!-- Benefits -->
  <text x="50" y="620" font-size="14" font-weight="bold" fill="#2c3e50">Benefits of Inheritance:</text>
  <text x="50" y="645" font-size="12" fill="#2c3e50">• Code reuse through base classes</text>
  <text x="50" y="665" font-size="12" fill="#2c3e50">• Polymorphic behavior - same interface, different implementations</text>
  <text x="450" y="645" font-size="12" fill="#2c3e50">• Consistent error handling across database types</text>
  <text x="450" y="665" font-size="12" fill="#2c3e50">• Easy extension with new database-specific exceptions</text>
</svg>