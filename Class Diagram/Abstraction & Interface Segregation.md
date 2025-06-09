svg viewBox=0 0 800 600 xmlns=httpwww.w3.org2000svg
  !-- Background --
  rect width=800 height=600 fill=#f8f9fa
  
  !-- Title --
  text x=400 y=30 text-anchor=middle font-size=20 font-weight=bold fill=#2c3e50Abstraction Through Interfacestext
  
  !-- Interface boxes --
  g
    !-- IDatabaseReader Interface --
    rect x=50 y=80 width=200 height=120 fill=#e8f4fd stroke=#3498db stroke-width=2 rx=5
    text x=150 y=100 text-anchor=middle font-weight=bold fill=#2c3e50«interface»text
    text x=150 y=120 text-anchor=middle font-weight=bold fill=#2c3e50IDatabaseReadertext
    line x1=70 y1=130 x2=230 y2=130 stroke=#3498db
    text x=70 y=150 font-size=12 fill=#2c3e50+ read_as_tuple()text
    text x=70 y=170 font-size=12 fill=#2c3e50+ read_as_dataframe()text
    
    !-- IDatabaseWriter Interface --
    rect x=300 y=80 width=200 height=120 fill=#e8f4fd stroke=#3498db stroke-width=2 rx=5
    text x=400 y=100 text-anchor=middle font-weight=bold fill=#2c3e50«interface»text
    text x=400 y=120 text-anchor=middle font-weight=bold fill=#2c3e50IDatabaseWritertext
    line x1=320 y1=130 x2=480 y2=130 stroke=#3498db
    text x=320 y=150 font-size=12 fill=#2c3e50+ execute_write()text
    text x=320 y=170 font-size=12 fill=#2c3e50+ execute_batch()text
    
    !-- IConnectionPool Interface --
    rect x=550 y=80 width=200 height=120 fill=#e8f4fd stroke=#3498db stroke-width=2 rx=5
    text x=650 y=100 text-anchor=middle font-weight=bold fill=#2c3e50«interface»text
    text x=650 y=120 text-anchor=middle font-weight=bold fill=#2c3e50IConnectionPooltext
    line x1=570 y1=130 x2=730 y2=130 stroke=#3498db
    text x=570 y=150 font-size=12 fill=#2c3e50+ get_connection()text
    text x=570 y=170 font-size=12 fill=#2c3e50+ return_connection()text
  g
  
  !-- Implementation boxes --
  g
    !-- DatabaseReader --
    rect x=50 y=280 width=200 height=100 fill=#e8f8e8 stroke=#27ae60 stroke-width=2 rx=5
    text x=150 y=305 text-anchor=middle font-weight=bold fill=#2c3e50DatabaseReadertext
    line x1=70 y1=315 x2=230 y2=315 stroke=#27ae60
    text x=70 y=335 font-size=12 fill=#2c3e50- connection_pooltext
    text x=70 y=355 font-size=12 fill=#2c3e50- error_handlertext
    
    !-- DatabaseWriter --
    rect x=300 y=280 width=200 height=100 fill=#e8f8e8 stroke=#27ae60 stroke-width=2 rx=5
    text x=400 y=305 text-anchor=middle font-weight=bold fill=#2c3e50DatabaseWritertext
    line x1=320 y1=315 x2=480 y2=315 stroke=#27ae60
    text x=320 y=335 font-size=12 fill=#2c3e50- connection_pooltext
    text x=320 y=355 font-size=12 fill=#2c3e50- error_handlertext
    
    !-- BasicConnectionPool --
    rect x=550 y=280 width=200 height=100 fill=#e8f8e8 stroke=#27ae60 stroke-width=2 rx=5
    text x=650 y=305 text-anchor=middle font-weight=bold fill=#2c3e50BasicConnectionPooltext
    line x1=570 y1=315 x2=730 y2=315 stroke=#27ae60
    text x=570 y=335 font-size=12 fill=#2c3e50- pool Queuetext
    text x=570 y=355 font-size=12 fill=#2c3e50- active_connectionstext
  g
  
  !-- Implementation arrows --
  defs
    marker id=triangle markerWidth=10 markerHeight=10 refX=9 refY=3 orient=auto markerUnits=strokeWidth
      polygon points=0,0 0,6 9,3 fill=#3498db
    marker
  defs
  
  line x1=150 y1=200 x2=150 y2=280 stroke=#3498db stroke-width=2 marker-end=url(#triangle) stroke-dasharray=5,5
  line x1=400 y1=200 x2=400 y2=280 stroke=#3498db stroke-width=2 marker-end=url(#triangle) stroke-dasharray=5,5
  line x1=650 y1=200 x2=650 y2=280 stroke=#3498db stroke-width=2 marker-end=url(#triangle) stroke-dasharray=5,5
  
  !-- Benefits text --
  text x=50 y=450 font-size=14 font-weight=bold fill=#2c3e50Benefits of Abstractiontext
  text x=50 y=480 font-size=12 fill=#2c3e50• Hides implementation complexitytext
  text x=50 y=500 font-size=12 fill=#2c3e50• Allows multiple implementationstext
  text x=50 y=520 font-size=12 fill=#2c3e50• Enables dependency injectiontext
  text x=50 y=540 font-size=12 fill=#2c3e50• Facilitates testing with mockstext
svg