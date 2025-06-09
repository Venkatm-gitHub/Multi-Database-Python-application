<svg viewBox="0 0 1000 800" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="1000" height="800" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="500" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Composition & Dependency Injection</text>
  
  <!-- DatabaseSession (Main Composite) -->
  <rect x="400" y="80" width="200" height="120" fill="#ff7675" stroke="#d63031" stroke-width="3" rx="5"/>
  <text x="500" y="105" text-anchor="middle" font-weight="bold" font-size="14" fill="white">DatabaseSession</text>
  <line x1="420" y1="115" x2="580" y2="115" stroke="white" stroke-width="2"/>
  <text x="420" y="135" font-size="11" fill="white">- _reader: IDatabaseReader</text>
  <text x="420" y="150" font-size="11" fill="white">- _writer: IDatabaseWriter</text>
  <text x="420" y="165" font-size="11" fill="white">- _transaction_manager</text>
  <text x="420" y="180" font-size="11" fill="white">- _connection_pool</text>
  
  <!-- Components that DatabaseSession contains -->
  <g>
    <!-- DatabaseReader -->
    <rect x="50" y="280" width="180" height="100" fill="#74b9ff" stroke="#0984e3" stroke-width="2" rx="5"/>
    <text x="140" y="305" text-anchor="middle" font-weight="bold" fill="white">DatabaseReader</text>
    <line x1="70" y1="315" x2="210" y2="315" stroke="white"/>
    <text x="70" y="335" font-size="11" fill="white">- connection_pool</text>
    <text x="70" y="350" font-size="11" fill="white">- error_handler</text>
    <text x="70" y="365" font-size="11" fill="white">- config</text>
    
    <!-- DatabaseWriter -->
    <rect x="270" y="280" width="180" height="100" fill="#00b894" stroke="#00a085" stroke-width="2" rx="5"/>
    <text x="360" y="305" text-anchor="middle" font-weight="bold" fill="white">DatabaseWriter</text>
    <line x1="290" y1="315" x2="430" y2="315" stroke="white"/>
    <text x="290" y="335" font-size="11" fill="white">- connection_pool</text>
    <text x="290" y="350" font-size="11" fill="white">- error_handler</text>
    <text x="290" y="365" font-size="11" fill="white">- config</text>
    
    <!-- TransactionManager -->
    <rect x="570" y="280" width="180" height="100" fill="#a29bfe" stroke="#6c5ce7" stroke-width="2" rx="5"/>
    <text x="660" y="305" text-anchor="middle" font-weight="bold" fill="white">TransactionManager</text>
    <line x1="590" y1="315" x2="730" y2="315" stroke="white"/>
    <text x="590" y="335" font-size="11" fill="white">- connection_pool</text>
    <text x="590" y="350" font-size="11" fill="white">- error_handler</text>
    
    <!-- BasicConnectionPool -->
    <rect x="770" y="280" width="180" height="100" fill="#fd79a8" stroke="#e84393" stroke-width="2" rx="5"/>
    <text x="860" y="305" text-anchor="middle" font-weight="bold" fill="white">BasicConnectionPool</text>
    <line x1="790" y1="315" x2="930" y2="315" stroke="white"/>
    <text x="790" y="335" font-size="11" fill="white">- pool: Queue</text>
    <text x="790" y="350" font-size="11" fill="white">- connection_factory</text>
    <text x="790" y="365" font-size="11" fill="white">- active_connections</text>
  </g>
  
  <!-- Composition relationships (filled diamonds) -->
  <defs>
    <marker id="composition" markerWidth="12" markerHeight="12" refX="12" refY="6" orient="auto">
      <polygon points="0,6 6,0 12,6 6,12" fill="#2c3e50"/>
    </marker>
    <marker id="dependency" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <polygon points="0,0 0,6 9,3" fill="none" stroke="#e74c3c" stroke-width="2"/>
    </marker>
  </defs>
  
  <!-- Composition arrows from DatabaseSession to its components -->
  <line x1="450" y1="200" x2="140" y2="280" stroke="#2c3e50" stroke-width="3" marker-end="url(#composition)"/>
  <line x1="480" y1="200" x2="360" y2="280" stroke="#2c3e50" stroke-width="3" marker-end="url(#composition)"/>
  <line x1="520" y1="200" x2="660" y2="280" stroke="#2c3e50" stroke-width="3" marker-end="url(#composition)"/>
  <line x1="550" y1="200" x2="860" y2="280" stroke="#2c3e50" stroke-width="3" marker-end="url(#composition)"/>
  
  <!-- Shared dependencies (connection pool used by multiple components) -->
  <line x1="230" y1="330" x2="770" y2="330" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#dependency)"/>
  <line x1="450" y1="330" x2="770" y2="330" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#dependency)"/>
  <line x1="660" y1="380" x2="770" y2="340" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#dependency)"/>
  
  <!-- Dependency Injection Container -->
  <g transform="translate(50, 450)">
    <rect x="0" y="0" width="900" height="200" fill="#f0f0f0" stroke="#bdc3c7" stroke-width="2" rx="10"/>
    <text x="20" y="25" font-size="16" font-weight="bold" fill="#2c3e50">Dependency Injection Container</text>
    
    <!-- Service creation flow -->
    <rect x="20" y="40" width="150" height="60" fill="#3498db" stroke="#2980b9" stroke-width="2" rx="5"/>
    <text x="95" y="65" text-anchor="middle" font-weight="bold" fill="white">DatabaseService</text>
    <text x="95" y="80" text-anchor="middle" font-size="11" fill="white">create_session()</text>
    
    <text x="190" y="75" font-size="20" fill="#2c3e50">→</text>
    
    <rect x="220" y="40" width="150" height="60" fill="#e67e22" stroke="#d35400" stroke-width="2" rx="5"/>
    <text x="295" y="60" text-anchor="middle" font-weight="bold" fill="white">Configuration</text>
    <text x="295" y="75" text-anchor="middle" font-size="11" fill="white">Provider</text>
    <text x="295" y="90" text-anchor="middle" font-size="11" fill="white">get_config()</text>
    
    <text x="390" y="75" font-size="20" fill="#2c3e50">→</text>
    
    <rect x="420" y="40" width="150" height="60" fill="#9b59b6" stroke="#8e44ad" stroke-width="2" rx="5"/>
    <text x="495" y="60" text-anchor="middle" font-weight="bold" fill="white">Factory</text>
    <text x="495" y="75" text-anchor="middle" font-size="11" fill="white">Registry</text>
    <text x="495" y="90" text-anchor="middle" font-size="11" fill="white">get_factory()</text>
    
    <text x="590" y="75" font-size="20" fill="#2c3e50">→</text>
    
    <rect x="620" y="40" width="150" height="60" fill="#27ae60" stroke="#229954" stroke-width="2" rx="5"/>
    <text x="695" y="65" text-anchor="middle" font-weight="bold" fill="white">Components</text>
    <text x="695" y="80" text-anchor="middle" font-size="11" fill="white">Creation</text>
    
    <!-- Benefits -->
    <text x="20" y="130" font-size="14" font-weight="bold" fill="#2c3e50">Benefits of Composition & DI:</text>
    <text x="30" y="155" font-size="12" fill="#2c3e50">• Loose coupling - components depend on abstractions</text>
    <text x="30" y="175" font-size="12" fill="#2c3e50">• Easy testing - can inject mock dependencies</text>
    <text x="450" y="155" font-size="12" fill="#2c3e50">• Flexible configuration - different implementations</text>
    <text x="450" y="175" font-size="12" fill="#2c3e50">• Single Responsibility - each class has one job</text>
  </g>
  
  <!-- Legend -->
  <g transform="translate(50, 720)">
    <text x="0" y="0" font-size="14" font-weight="bold" fill="#2c3e50">Legend:</text>
    <line x1="0" y1="15" x2="30" y2="15" stroke="#2c3e50" stroke-width="3" marker-end="url(#composition)"/>
    <text x="40" y="20" font-size="12" fill="#2c3e50">Composition (has-a)</text>
    <line x1="200" y1="15" x2="230" y2="15" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#dependency)"/>
    <text x="240" y="20" font-size="12" fill="#2c3e50">Dependency (uses)</text>
  </g>
</svg>