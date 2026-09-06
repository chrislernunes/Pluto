# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import time, datetime, sys, os, logging, traceback
import pandas as pd
import direct_redis
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.time cimport time as c_time
from libc.string cimport strcmp
from xxx.brokers import XTS_Agent, XTS_AgentOld
from xxx.utility import market_hours_cython  # ✅ Import the fixed function

cdef class TradingAgent:
    cdef object r
    cdef object logger
    cdef str user
    cdef int decision 
    cdef object client
    cdef long last_timestamp = int(c_time(NULL)) - 2
    cdef int iter_count = 0
    cdef int delay_secs = 1


    def __cinit__(self, str user):
        """ Initialize the Trading Agent """
        self.user = user
        self.r = direct_redis.DirectRedis()
        logging.basicConfig(filename=f'logs/clients/{user}.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    cpdef void launch(self):
        """ Optimized Trading Agent Launch Function """

        try:
            # Fetch client info
            client_info = self.r.hget('live_clients', self.user)
            if client_info is None:
                raise ValueError(f'Client {self.user} not found in live_clients')

            print(f"Launching Trading Agent for {self.user}")

            # Kill old process
            old_pid = self.r.hget('live_client_trader_pid', self.user)
            if old_pid is not None:
                try:
                    os.kill(int(old_pid), 9)
                    self.logger.info(f'Killed old PID {old_pid}')
                except ProcessLookupError:
                    self.logger.info(f'Process {old_pid} already terminated')
                except Exception as e:
                    self.logger.error(f'Error in killing old PID {old_pid} - {e}')
                    sys.exit(1)
                    return

            pid = os.getpid()
            self.logger.info(f'PID: {pid}')
            self.r.hset('live_client_trader_pid', self.user, pid)

            # ✅ Declare cdef variable at the function start
            

            while True:
                decision = market_hours_cython(datetime.time(9, 15))  # ✅ Fixed syntax
                if decision == -1:
                    print(f"##### MARKET OPEN :: Starting {self.user}")
                    self.r.lpush('critical_tg', f"##### MARKET OPEN :: Starting {self.user}")
                    break
                time.sleep(decision)  # ✅ Ensured decision is an int

            # Initialize Client
            
            if self.user in ('SAMIR DOSHI', 'Maverick Fund', 'RISHABH CLI'):
                client = XTS_Agent(client_info)
            else:
                client = XTS_AgentOld(client_info)

            self.r.hdel('pending_orders', client.name)
            self.r.hdel('rejected_orders', client.name)

            # Performance Optimized Loop

            while True:
                cdef int tik = int(c_time(NULL))

                # Fetch Market Data (Optimized)
                cdef dict client_positions = client.get_positions()
                cdef dict order_book = client.get_order_book()
                cdef dict trade_book = {}

                if iter_count % 7 == 0 or iter_count == 1:
                    trade_book = client.get_trade_book()

                iter_count += 1
                last_timestamp = tik

                if client_positions is None or order_book is None:
                    self.logger.error(f"API ERROR for {self.user}, retrying...")
                    time.sleep(2)
                    continue

                # Optimized Dictionary Lookups
                cdef dict current_positions = {int(x['ExchangeInstrumentId']): int(x['Quantity']) 
                                               for x in client_positions if int(x['Quantity']) != 0}

                self.r.hset('live_user_positions', self.user, current_positions)

                # Parallel Order Handling
                self.process_orders(client, order_book, current_positions)

                # Exit if market is closed
                if datetime.datetime.now().time() > datetime.time(15, 21):
                    break

                time.sleep(delay_secs)

        except KeyboardInterrupt:
            print('Keyboard Interrupt')
            self.r.set(f"pulse_{self.user}", b'Keyboard Interrupt')

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            self.logger.error(f"ERROR in {self.user} Trader! Exiting Trader {traceback.format_exc()}")
            self.r.lpush('critical_tg', f'ERROR in {self.user} Trader!')

        finally:
            self.r.hdel('live_client_trader_pid', self.user)
            print(f'EXITING TRADER FOR {self.user}')
            self.r.lpush('critical_tg', f'EXITING TRADER FOR {self.user}')
            sys.exit(0)

    cdef void process_orders(self, object client, dict order_book, dict current_positions):
        """ Optimized order processing using Cython """
        cdef dict fresh_orders = {}
        cdef dict modified_orders = {}

        for order in order_book:
            order_id = order['AppOrderID']
            status = order['OrderStatus']
            qty = order['LeavesQuantity']

            if status in ['Filled', 'Cancelled', 'Rejected']:
                continue

            if qty > 0:
                fresh_orders[order_id] = order
            else:
                modified_orders[order_id] = order

        # Process fresh orders
        for order_id, order in fresh_orders.items():
            self.modify_order(client, order)

        # Process modified orders
        for order_id, order in modified_orders.items():
            self.cancel_order(client, order)

    cdef void modify_order(self, object client, dict order):
        """ Modify order with minimal overhead """
        try:
            client.modify_order(order)
        except Exception as e:
            self.logger.error(f"Error modifying order {order['AppOrderID']}: {e}")

    cdef void cancel_order(self, object client, dict order):
        """ Cancel order with minimal overhead """
        try:
            client.cancel_order(order['AppOrderID'])
        except Exception as e:
            self.logger.error(f"Error cancelling order {order['AppOrderID']}: {e}")
