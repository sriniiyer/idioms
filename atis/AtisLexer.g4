lexer grammar AtisLexer;

LAMBDA:           'lambda';
E:           'e';
I:           'i';
ARGMAX:           'argmax';
ARGMIN:           'argmin';
EXISTS:           'exists';
EQUALS:           'equals';
MAX:           'max';
MIN:           'min';
EQUALTO:           '=';
SUM:           'sum';
GREATER:           '>';
LESSER:           '<';
COUNT:           'count';
AND:           'and';
OR:           'or';
THE:           'the';
NOT:           'not';
VARIABLE:     '$' [0-9];
PREDICATE:   'airport'|'oneway'|'flight_number'|'flight'|'airline'|'ground_transport'|'to_city'|'to'|'from'|'loc:t'|'rapid_transit'|'equals:t'|'day'|'fare'|'services'|'rental_car'|'abbrev'|'fare_basis_code'|'city'|'stop'|'day_number'|'nonstop'|'aircraft_code'|'count'|'limousine'|'airline:e'|'month'|'connecting'|'class_type'|'during_day'|'economy'|'named'|'from_airport'|'departure_time'|'weekday'|'capacity'|'minimum_connection_time'|'tomorrow'|'airline_name'|'aircraft_code:t'|'meal'|'meal_code'|'meal:t'|'daily'|'time_elapsed'|'time_zone_code'|'booking_class:t'|'arrival_time'|'booking_class'|'round_trip'|'has_meal'|'approx_arrival_time'|'approx_departure_time'|'year'|'minutes_distant'|'stops'|'during_day_arrival'|'day_arrival'|'next_days'|'day_number_arrival'|'month_arrival'|'aircraft'|'ground_fare'|'has_stops'|'after_day'|'manufacturer'|'day_after_tomorrow'|'stop_arrival_time'|'arrival_month'|'overnight'|'restriction_code'|'today'|'turboprop'|'miles_distant'|'taxi'|'jet'|'air_taxi_operation'|'before_day'|'tonight'|'tomorrow_arrival'|'day_number_return'|'days_from_today'|'approx_return_time'|'class_of_service'|'month_return'|'day_return'|'discounted'|'cost';
ARGUMENT:    [a-zA-Z]+ [0-9]+;
CONSTANT:    [a-zA-Z:_0-9]+;
LPAREN:             '(';
RPAREN:             ')';

WS:                 [ \t\r\n\u000C]+ -> channel(HIDDEN);
