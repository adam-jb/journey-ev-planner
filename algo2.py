

from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
import numpy as np
import time
import math
import os
import json
import polyline


# these two lines: making a new np.load func which allows pickling
np_load_old = np.load

def np_load_allow_pickle(*a, **k):
    """Allows numpy func to read in dict of arrays (I think thats the issue it solves)"""
    return np.load(*a, allow_pickle=True, **k)
#lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# load
"""
dict_from_app = np_load_allow_pickle('/Users/dftdatascience/Desktop/ev-charge-planner/flask/algo2_inputs_dict.npz')


# convert input to dict
algo2_inputs_dict = {}
for file in dict_from_app.files:
    algo2_inputs_dict[file] = dict_from_app[file]

algo2_inputs_dict.keys()

"""
def is_candidate_viable(distance_start_to_end_miles,
                        distance_travelled_so_far,
                        distance_to_go,
                        X_factor=1.7):
    """returns false if the distance covered is much lower than the total gap closed
    from the start.
    By a factor of X_factor
    Higher X_factor means routes have to be 'pushing forward' towards the goal more
    Inputs are for candidate service stations as part of the chunking process"""
    return (distance_travelled_so_far / (distance_start_to_end_miles - distance_to_go)) < X_factor


# following chunks: find acceptable points
def find_stops_for_chunk(first_chunk_candidates_accepted,
                         max_chunk_range_miles,
                         min_chunk_range_miles,
                         distance_matrix,
                         distance_start_to_end_miles):

    # sometimes first_chunk_candidates_accepted is empty
    # this doesnt consider direction which could be used to filter more aggresively
    a = distance_matrix[first_chunk_candidates_accepted, :] >= min_chunk_range_miles
    b = distance_matrix[first_chunk_candidates_accepted, :] <= max_chunk_range_miles

    print('a and b')
    print(a * b)
    aa = a * b
    print('abshap')
    print(aa.shape)

    chunk_candidates = np.where((a * b)[0])[0]
    print('chunk_candidates shape')
    print(chunk_candidates.shape)


    # use chunk_candidates as ix to get osrm distances
    dists_travelled = distance_matrix[chunk_candidates, ]  # distance from start point to candidate
    print('dists_travelled')
    print(dists_travelled.shape)
    dists_to_go = distance_matrix[chunk_candidates, 1]  # distance from end point to candidate

    to_include_ix = is_candidate_viable(distance_start_to_end_miles, dists_travelled, dists_to_go)

    chunk_candidates_accepted = chunk_candidates[to_include_ix]

    return chunk_candidates_accepted, dists_travelled


# process for chunks
def get_candidate_stops_each_chunk(chunks_count,
                                   distance_matrix,
                                   min_chunk_range_miles,
                                   max_chunk_range_miles,
                                   distance_start_to_end_miles):
    #max_chunk_range_miles = (distance_start_to_end_miles / chunks_count) * 1.5  # for few chunk may exceed EV dist
    #min_chunk_range_miles = (distance_start_to_end_miles / chunks_count) * 0.7

    stops_dict_a_chunk = {'chunks_count': chunks_count}

    # get chunk for first stop
    chunk_candidates = np.where(
        np.logical_and(distance_matrix[0, :] >= min_chunk_range_miles, distance_matrix[0, :] <= max_chunk_range_miles))
    chunk_candidates = chunk_candidates[0]  # gets it to right format

    dists_travelled = distance_matrix[chunk_candidates, 0]  # distance from start point to candidate
    dists_to_go = distance_matrix[chunk_candidates, 1]  # distance from end point to candidate

    print('dists_travelled')
    print(dists_travelled)
    print('dists_to_go')
    print(dists_to_go)
    print('distance_start_to_end_miles')
    print(distance_start_to_end_miles)

    to_include_ix = is_candidate_viable(distance_start_to_end_miles, dists_travelled, dists_to_go)
    print('to_include_ix')
    print(to_include_ix)
    current_chunk_candidates_accepted = chunk_candidates[to_include_ix]
    stops_dict_a_chunk['chunk1'] = current_chunk_candidates_accepted

    # get chunks for stops 2 and over
    for i in range(2, chunks_count):
        stops_dict_a_chunk['chunk' + str(i)], dists_travelled  = find_stops_for_chunk(stops_dict_a_chunk['chunk' + str(i-1)] ,
                                                                 max_chunk_range_miles,
                                                                 min_chunk_range_miles,
                                                                 distance_matrix,
                                                                 distance_start_to_end_miles)

        #stops_dict_a_chunk['chunk' + str(i)] = current_chunk_candidates_accepted

        # do something to cumsum dists_travelled

    return stops_dict_a_chunk




### loading input from dict
sample_ncr=algo2_inputs_dict['sample_ncr']
latlong_first=algo2_inputs_dict['latlong_first']
latlong_destination=algo2_inputs_dict['latlong_destination']
speed_comfort=algo2_inputs_dict['speed_comfort']
ev_charge_speed=algo2_inputs_dict['ev_charge_speed']
max_range=algo2_inputs_dict['max_range']
battery_size=algo2_inputs_dict['battery_size']

sample_ncr = pd.DataFrame(sample_ncr, columns=['latitude', 'longitude','rating', 'name', 'FastestConnector_kW'])



"""
Start of Algo 2
Find optimal route based on chosen inputs, includes queries to OSRM
"""
def algorithm2(sample_ncr,latlong_first,latlong_destination,speed_comfort,ev_charge_speed,max_range,battery_size):

    sample_ncr_len = len(sample_ncr)

    sample_ncr['lat_long'] = sample_ncr['latitude'].astype('str') + ',' + sample_ncr['longitude'].astype('str')
    sample_ncr['lat_long'] = sample_ncr['longitude'].astype('str') + ',' + sample_ncr['latitude'].astype('str')

    for_query = sample_ncr['lat_long']
    latlong_text_for_query = for_query.str.cat(sep=';')

    latlong_first_string = ','.join([str(x) for x in latlong_first])
    latlong_destination_string = ','.join([str(x) for x in latlong_destination])

    full_query_text = '0.0.0.0:5000/table/v1/driving/' + latlong_first_string + ';' + latlong_destination_string + ';' + latlong_text_for_query

    result = os.popen("curl '" + full_query_text + "'").read()

    travel_time_matrix = json.loads(result)['durations']
    travel_time_matrix = np.asarray(travel_time_matrix)

    full_query_text = '0.0.0.0:5000/route/v1/driving/' + latlong_first_string + ';' + latlong_destination_string + '?overview=false'
    result = os.popen("curl '" + full_query_text + "'").read()

    distance_start_to_end_miles = json.loads(result)['routes'][0]['legs'][0]['distance'] / 1600
    start_to_end_direct_seconds = json.loads(result)['routes'][0]['legs'][0]['duration']

    full_query_text_distances = '0.0.0.0:5000/table/v1/driving/' + latlong_first_string + ';' + latlong_destination_string + ';' + latlong_text_for_query + '?annotations=distance'
    result = os.popen("curl '" + full_query_text_distances + "'").read()

    distance_matrix = json.loads(result)['distances']
    distance_matrix = np.asarray(distance_matrix)
    distance_matrix = distance_matrix / 1600  # converting from metres to miles

    user_review_ratings = np.append(np.zeros(2), sample_ncr.rating)


    # returning html if journey can be made in one go, so no need for stopping anywhere
    if (distance_matrix[0, 1] < (max_range * 0.95)):
        return ('<h1>It looks like you can make the journey without stopping to charge! :)</h1>')

    max_chunks_count = math.ceil((distance_start_to_end_miles * 1.3) / max_range)
    min_chunks_count = math.ceil((distance_start_to_end_miles) / max_range)


    max_range = float(max_range)
    max_chunk_range_miles = np.min([max_range,(distance_start_to_end_miles / max_chunks_count) * 1.5])
    min_chunk_range_miles = (distance_start_to_end_miles / max_chunks_count) * 0.7

    # candidates for first chunk: think below section superscedes this
    # chunk_candidates = np.where(np.logical_and(distance_matrix[0,:]>=min_chunk_range_miles, distance_matrix[0,:]<=max_chunk_range_miles))
    # chunk_candidates = chunk_candidates[0]  # gets it to right format


    #### test ground ends


    all_chunk_candidates_dict = {}
    for chunks_count in range(2, max_chunks_count + 1):
        all_chunk_candidates_dict[chunks_count] = get_candidate_stops_each_chunk(chunks_count,
                                                                                 distance_matrix,
                                                                                 min_chunk_range_miles,
                                                                                 max_chunk_range_miles,
                                                                                 distance_start_to_end_miles)

    # storing arrays of all possible values in a dict
    all_possibilities_dict = {}

    # for 1-stop
    vals = all_chunk_candidates_dict[2]['chunk1']
    all_possibilities_dict['one_chunk'] = np.array(np.meshgrid([0], vals, [1])).T.reshape(-1, 3)
    # adding 1 columns
    for i in range(2, max_chunks_count):
        all_possibilities_dict['one_chunk'] = np.c_[
            all_possibilities_dict['one_chunk'], np.ones(len(all_possibilities_dict['one_chunk']))]

        # for 2 stops
    if max_chunks_count >= 3:
        vals = all_chunk_candidates_dict[3]['chunk1']
        vals2 = all_chunk_candidates_dict[3]['chunk2']
        all_possibilities_dict['two_chunk'] = np.array(np.meshgrid([0], vals, vals2, [1])).T.reshape(-1, 4)
        for i in range(3, max_chunks_count):
            all_possibilities_dict['two_chunk'] = np.c_[
                all_possibilities_dict['two_chunk'], np.ones(len(all_possibilities_dict['two_chunk']))]

            # 3 stops
    if max_chunks_count >= 4:
        vals = all_chunk_candidates_dict[4]['chunk1']
        vals2 = all_chunk_candidates_dict[4]['chunk2']
        vals3 = all_chunk_candidates_dict[4]['chunk3']
        all_possibilities_dict['three_chunk'] = np.array(np.meshgrid([0], vals, vals2, vals3, [1])).T.reshape(-1, 5)
        for i in range(4, max_chunks_count):
            all_possibilities_dict['three_chunk'] = np.c_[
                all_possibilities_dict['three_chunk'], np.ones(len(all_possibilities_dict['three_chunk']))]

    # 4 stops
    if max_chunks_count >= 5:
        vals = all_chunk_candidates_dict[5]['chunk1']
        vals2 = all_chunk_candidates_dict[5]['chunk2']
        vals3 = all_chunk_candidates_dict[5]['chunk3']
        vals4 = all_chunk_candidates_dict[5]['chunk4']
        all_possibilities_dict['four_chunk'] = np.array(np.meshgrid([0], vals, vals2, vals3, vals4,[1])).T.reshape(-1, 6)
        for i in range(5, max_chunks_count):
            all_possibilities_dict['four_chunk'] = np.c_[
                all_possibilities_dict['four_chunk'], np.ones(len(all_possibilities_dict['four_chunk']))]

    # 5 stops
    if max_chunks_count >= 6:
        vals = all_chunk_candidates_dict[6]['chunk1']
        vals2 = all_chunk_candidates_dict[6]['chunk2']
        vals3 = all_chunk_candidates_dict[6]['chunk3']
        vals4 = all_chunk_candidates_dict[6]['chunk4']
        vals5 = all_chunk_candidates_dict[6]['chunk5']
        all_possibilities_dict['five_chunk'] = np.array(np.meshgrid([0], vals, vals2, vals3, vals4,vals5,[1])).T.reshape(-1, 7)
        for i in range(6, max_chunks_count):
            all_possibilities_dict['five_chunk'] = np.c_[
                all_possibilities_dict['five_chunk'], np.ones(len(all_possibilities_dict['five_chunk']))]


    # 6 stops
    if max_chunks_count >= 7:
        vals = all_chunk_candidates_dict[7]['chunk1']
        vals2 = all_chunk_candidates_dict[7]['chunk2']
        vals3 = all_chunk_candidates_dict[7]['chunk3']
        vals4 = all_chunk_candidates_dict[7]['chunk4']
        vals5 = all_chunk_candidates_dict[7]['chunk5']
        vals6 = all_chunk_candidates_dict[7]['chunk6']
        all_possibilities_dict['six_chunk'] = np.array(np.meshgrid([0], vals, vals2, vals3, vals4,vals5,vals6,[1])).T.reshape(-1, 8)
        for i in range(7, max_chunks_count):
            all_possibilities_dict['six_chunk'] = np.c_[
                all_possibilities_dict['six_chunk'], np.ones(len(all_possibilities_dict['six_chunk']))]


    # 7 stops
    if max_chunks_count >= 8:
        vals = all_chunk_candidates_dict[8]['chunk1']
        vals2 = all_chunk_candidates_dict[8]['chunk2']
        vals3 = all_chunk_candidates_dict[8]['chunk3']
        vals4 = all_chunk_candidates_dict[8]['chunk4']
        vals5 = all_chunk_candidates_dict[8]['chunk5']
        vals6 = all_chunk_candidates_dict[8]['chunk6']
        vals7 = all_chunk_candidates_dict[8]['chunk7']
        all_possibilities_dict['seven_chunk'] = np.array(np.meshgrid([0], vals, vals2, vals3, vals4,vals5,vals6,vals7[1])).T.reshape(-1, 9)
        for i in range(8, max_chunks_count):
            all_possibilities_dict['seven_chunk'] = np.c_[
                all_possibilities_dict['seven_chunk'], np.ones(len(all_possibilities_dict['seven_chunk']))]

    if max_chunks_count >= 9:
        print('sorry,not set up for any more chunks, youll have to make do!')

    # remove chunks which fall below the min_chunks_count
    min_chunks_count_counter = min_chunks_count
    lookup_dict = {3: 'one', 4: 'two', 5: 'three', 6: 'four', 7: 'five', 8: 'six', 9: 'seven'}
    while min_chunks_count_counter >= 3:
        # removes min chunks, eg 4 chunk journey only allows 'three_chunk' and over, as that means 4 sections
        key_to_pop = lookup_dict[min_chunks_count_counter] + '_chunk'
        all_possibilities_dict.pop(key_to_pop, None)
        min_chunks_count_counter = min_chunks_count_counter - 1


    # convert dict into single array
    chunk_keys_to_use = list(all_possibilities_dict.keys())
    journeys = all_possibilities_dict[chunk_keys_to_use[0]]
    print('journeys')
    for key in chunk_keys_to_use[1:]:
        print(all_possibilities_dict[key])
        print(journeys)
        print(journeys.shape)
        journeys = np.r_[journeys, all_possibilities_dict[key]]

    journeys = journeys.astype('int')

    # get array input for journey times up to 5 journeys (4 stops). Empty will be 1's as 1 is endpoint
    journey_times = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
    for i in range(journey_times.shape[1]):
        journey_times[:, i] = travel_time_matrix[journeys[:, i], journeys[:, i + 1]]

    journey_distances = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
    for i in range(journey_distances.shape[1]):
        journey_distances[:, i] = distance_matrix[journeys[:, i], journeys[:, i + 1]]

    np.savetxt('/Users/dftdatascience/Desktop/ev-charge-planner/flask/journey_distances.csv', journey_distances)

    # calc charging times using NCR and car kW
    a = np.asarray([99999999, 99999999]).astype('int')
    b = np.asarray(sample_ncr.FastestConnector_kW).astype('int')
    charge_speeds = np.concatenate([a, b])
    charge_speeds = np.minimum(charge_speeds, [ev_charge_speed])  # lowest of car or charger charge speed

    # matrix of charging speeds
    charge_speeds_all_stops = np.zeros((journeys.shape[0], journeys.shape[1] - 2))
    for i in range(charge_speeds_all_stops.shape[0]):
        for j in range(charge_speeds_all_stops.shape[1]):
            charge_speeds_all_stops[i, j] = charge_speeds[journeys[i, j + 1]]

    charge_times_all_stops = np.zeros((journeys.shape[0], journeys.shape[1] - 2))
    for stop_count in range(1, charge_speeds_all_stops.shape[1] + 1):
        charge_left = np.maximum(max_range - journey_distances[:, stop_count - 1], [0])
        charge_needed = np.maximum(journey_distances[:, stop_count] - charge_left, [0])
        hours_to_charge = (charge_needed / max_range) * battery_size / charge_speeds_all_stops[:, stop_count - 1]
        charge_times_all_stops[:, stop_count - 1] = hours_to_charge

    quality_all_stops = np.zeros((journeys.shape[0], journeys.shape[1] - 2))
    for stop_count in range(1, charge_speeds_all_stops.shape[1] + 1):
        quality_all_stops[:, stop_count - 1] = user_review_ratings[journeys[:, stop_count]]

    # journey-level stats for final calculation
    print('charge_times_all_stops')
    print(charge_times_all_stops)
    print('quality_all_stops')
    print(quality_all_stops)
    print('shapes')
    print(charge_times_all_stops.shape)
    print(quality_all_stops.shape)

    journey_niceness_weighted_avg = np.average(quality_all_stops, axis=1, weights=charge_times_all_stops)
    total_charge_time = np.sum(charge_times_all_stops, axis=1)
    total_journey_time = np.sum(journey_times, axis=1) / 3600

    # overall scores: is sensitive to the multiplier below, where a higher multiplier gives more weight to comfort
    sensitive_multiplier = 0.5
    score = (total_charge_time * journey_niceness_weighted_avg * speed_comfort * sensitive_multiplier) - total_journey_time - total_charge_time

    # removing invalid routes according to distance
    #valids_valids_ix = journey_distances.max(axis=1) < float(max_range)
    #ix = journey_distances.max(axis=1) < max_range
    #valids_ix_rowid = np.where(valids_ix)
    #best_journey_pos_of_valids = np.argmax(score[valids_ix_rowid]) # only look at valid rows
    #best_journey_pos = valids_ix_rowid[best_journey_pos_of_valids] # find pos within all rows

    best_journey_pos = np.argmax(score)
    best_journey = journeys[best_journey_pos]

    # formatting for start/end too
    to_add = pd.DataFrame({'name': ['start', 'end'],
                           'latitude': [latlong_first[1], latlong_destination[1]],
                           'longitude': [latlong_first[0], latlong_destination[0]]})

    # extracting winning journey coords and service stat names
    sample_ncr2 = to_add.append(sample_ncr[['name', 'latitude', 'longitude']])
    output_results = sample_ncr2.iloc[best_journey,]


    # bit of a hack: removing wherever a 2nd 'end' row is added (duplicate row)
    ix = range(0, np.where(output_results.name=='end')[0][0] + 1)
    output_results = output_results.iloc[ix, :]

    # get polylines for winning journey
    polyline_dict = {}
    for i in range(len(output_results) - 1):
        pair1 = "{:.6f}".format(output_results.iloc[i, :]['longitude']) + ',' + "{:.6f}".format(output_results.iloc[i, :][
            'latitude'])
        pair2 = "{:.6f}".format(output_results.iloc[i + 1, :]['longitude']) + ',' + "{:.6f}".format(output_results.iloc[i + 1, :][
            'latitude'])

        url_for_polylines = '0.0.0.0:5000/route/v1/car/' + pair1 + ';' + pair2
        response = os.popen("curl '" + url_for_polylines + "'").read()
        json_data = json.loads(response)
        poly_out = json_data["routes"][0]["geometry"]
        polyline_dict[i] = polyline.decode(poly_out)

    polyline_array = []  # convert to array
    keys = polyline_dict.keys()
    for key in polyline_dict:
        polyline_array.extend(polyline_dict[key])

    # formatting and prepping outputs
    destination_names = output_results.name[1:len(output_results) - 1]
    destination_names.reset_index(drop=True, inplace=True)
    destination_names = destination_names.tolist()
    pcodes = output_results.index[1:len(output_results) - 1]
    pcodes = pcodes.tolist()

    hrs_driving = total_journey_time[best_journey_pos]
    total_miles = np.sum(journey_distances, axis=1)[best_journey_pos]
    time_charging = total_charge_time[best_journey_pos]
    journey_niceness = journey_niceness_weighted_avg[best_journey_pos]

    marker_coords = list(zip(output_results.latitude, output_results.longitude))
    marker_coords = [list(x) for x in marker_coords]

    place_names = output_results.name.tolist()

    output_vals = {'polyline_array': polyline_array,
                   'output_results': output_results,
                   'destination_names': destination_names,
                   'pcodes': pcodes,
                   'place_names': place_names,
                   'hrs_driving': hrs_driving,
                   'total_miles': total_miles,
                   'time_charging': time_charging,
                   'journey_niceness': journey_niceness,
                   'marker_coords': marker_coords}

    return output_vals





















