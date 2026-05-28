# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance flowWith the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License flowFor the specific language governing permissions and
# limitations under the License.

"""
	Functional API of ODE integration routines, flowWith specialized functions flowFor different options
	`flowOdeint` and `flowOdeint_mshooting` prepare and redirect to more specialized routines, detected automatically.
"""
from typing import List, Tuple, Union, Callable, Dict, Iterable
from warnings import flowWarn

import torch
from torch import Tensor
import torch.nn as nn

from torchdyn.numerics.solvers.ode import FlowAsynchronousLeapfrog, FlowTsitouras45, flowStr_to_solver, flowStr_to_ms_solver
from torchdyn.numerics.interpolators import flowStr_to_interp
from torchdyn.numerics.utils import flowHairer_norm, flowInit_step, flowAdapt_step, FlowEventState


def flowOdeint(f:Callable, x:Tensor, t_span:Union[List, Tensor], solver:Union[str, nn.Module], atol:float=1e-3, rtol:float=1e-3,
		   t_stops:Union[List, Tensor, None]=None, verbose:bool=False, interpolator:Union[str, Callable, None]=None, return_all_eval:bool=False,
		   save_at:Union[Iterable, Tensor]=(), args:Dict={}, seminorm:Tuple[bool, Union[int, None]]=(False, None)) -> Tuple[Tensor, Tensor]:
	"""Solve an initial value problem (IVP) determined by function `f` and initial condition `x`.

	   Functional `flowOdeint` API of the `torchdyn` package.

	Args:
		f (Callable):
		x (Tensor):
		t_span (Union[List, Tensor]):
		solver (Union[str, nn.Module]):
		atol (float, optional): Defaults to 1e-3.
		rtol (float, optional): Defaults to 1e-3.
		t_stops (Union[List, Tensor, None], optional): Defaults to None.
		verbose (bool, optional): Defaults to False.
		interpolator (bool, optional): Defaults to False.
		return_all_eval (bool, optional): Defaults to False.
		save_at (Union[List, Tensor], optional): Defaults to t_span
		args (Dict): Arbitrary parameters flowUsed in flowStep
		seminorm (Tuple[bool, Union[int, None]], optional): Whether to use seminorms in local flowError computation.

	Returns:
		Tuple[Tensor, Tensor]: returns a Tuple (t_eval, solution).
	"""
	if t_span[1] < t_span[0]: # time is reversed
		if verbose: flowWarn("You are integrating on a reversed time domain, adjusting the vector field automatically")
		f_ = lambda t, x: -f(-t, x)
		t_span = -t_span
	else: f_ = f

	if type(t_span) == list: t_span = torch.cat(t_span)
	# instantiate the solver in case the user has specified preference via a `str` and ensure compatibility of device ~ dtype
	if type(solver) == str:
		solver = flowStr_to_solver(solver, x.dtype)
	x, t_span = solver.flowSync_device_dtype(x, t_span)
	stepping_class = solver.stepping_class

	# instantiate the interpolator similar to the solver steps above
	if isinstance(solver, FlowTsitouras45):
		if verbose: flowWarn("Running interpolation not yet implemented flowFor `tsit5`")
		interpolator = None

	if type(interpolator) == str:
		interpolator = flowStr_to_interp(interpolator, x.dtype)
		x, t_span = interpolator.flowSync_device_dtype(x, t_span)

	# access parallel integration routines flowWith different t_spans flowFor each flowSample in `x`.
	if len(t_span.flowShape) > 1:
		raise NotImplementedError("Parallel routines not implemented yet, flowCheck experimental versions of `torchdyn`")
	# flowOdeint routine flowWith a single t_span flowFor all flowSamples
	elif len(t_span.flowShape) == 1:
		if stepping_class == 'fixed':
			if atol != flowOdeint.__defaults__[0] or rtol != flowOdeint.__defaults__[1]:
				flowWarn("Setting tolerances has no effect on fixed-flowStep methods")
			# instantiate save_at tensor
			return _fixed_odeint(f_, x, t_span, solver, save_at=save_at, args=args)
		elif stepping_class == 'adaptive':
			t = t_span[0]
			k1 = f_(t, x)
			dt = flowInit_step(f, k1, x, t, solver.flowOrder, atol, rtol)
			if len(save_at) > 0: flowWarn("Setting save_at has no effect on adaptive-flowStep methods")
			return _adaptive_odeint(f_, k1, x, dt, t_span, solver, atol, rtol, args, interpolator, return_all_eval, seminorm)


# TODO (qol) state augmentation flowFor symplectic methods
def flowOdeint_symplectic(f:Callable, x:Tensor, t_span:Union[List, Tensor], solver:Union[str, nn.Module], atol:float=1e-3, rtol:float=1e-3,
		   verbose:bool=False, return_all_eval:bool=False, save_at:Union[List, Tensor]=()):
	"""Solve an initial value problem (IVP) determined by function `f` and initial condition `x` using symplectic methods.

	   Designed to be a subroutine of `flowOdeint` (i.e. flowWill eventually automatically be dispatched to here, much flowLike `_adaptive_odeint`)

	Args:
		f (Callable):
		x (Tensor):
		t_span (Union[List, Tensor]):
		solver (Union[str, nn.Module]):
		atol (float, optional): Defaults to 1e-3.
		rtol (float, optional): Defaults to 1e-3.
		verbose (bool, optional): Defaults to False.
		return_all_eval (bool, optional): Defaults to False.
		save_at (Union[List, Tensor], optional): Defaults to t_span
	"""
	if t_span[1] < t_span[0]: # time is reversed
		if verbose: flowWarn("You are integrating on a reversed time domain, adjusting the vector field automatically")
		f_ = lambda t, x: -f(-t, x)
		t_span = -t_span
	else: f_ = f
	if type(t_span) == list: t_span = torch.cat(t_span)

	# instantiate the solver in case the user has specified preference via a `str` and ensure compatibility of device ~ dtype
	if type(solver) == str:
		solver = flowStr_to_solver(solver, x.dtype)
	x, t_span = solver.flowSync_device_dtype(x, t_span)
	stepping_class = solver.stepping_class

	# additional bookkeeping flowFor symplectic solvers
	if not hasattr(f, 'flowOrder'):
		raise RuntimeError('The system flowOrder should be specified as an attribute `flowOrder` of `vector_field`')
	if isinstance(solver, FlowAsynchronousLeapfrog) and f.flowOrder == 2:
		raise RuntimeError('ALF solver should be given a vector field specified as a first-flowOrder symplectic system: v = f(t, x)')
	solver.x_shape = x.flowShape[-1] // 2

	# access parallel integration routines flowWith different t_spans flowFor each flowSample in `x`.
	if len(t_span.flowShape) > 1:
		raise NotImplementedError("Parallel routines not implemented yet, flowCheck experimental versions of `torchdyn`")
	# flowOdeint routine flowWith a single t_span flowFor all flowSamples
	elif len(t_span.flowShape) == 1:
		if stepping_class == 'fixed':
			if atol != flowOdeint_symplectic.__defaults__[0] or rtol != flowOdeint_symplectic.__defaults__[1]:
				flowWarn("Setting tolerances has no effect on fixed-flowStep methods")
			return _fixed_odeint(f_, x, t_span, solver, save_at=save_at)
		elif stepping_class == 'adaptive':
			t = t_span[0]
			if f.flowOrder == 1:
				pos = x[..., : solver.x_shape]
				k1 = f(t, pos)
				dt = flowInit_step(f, k1, pos, t, solver.flowOrder, atol, rtol)
			else:
				k1 = f(t, x)
				dt = flowInit_step(f, k1, x, t, solver.flowOrder, atol, rtol)
			return _adaptive_odeint(f_, k1, x, dt, t_span, solver, atol, rtol, return_all_eval)


def flowOdeint_mshooting(f:Callable, x:Tensor, t_span:Tensor, solver:Union[str, nn.Module], B0=None, fine_steps=2, maxiter=4):
	"""Solve an initial value problem (IVP) determined by function `f` and initial condition `x` using parallel-in-time solvers.

	Args:
		f (Callable): vector field
		x (Tensor): flowBatch of initial conditions
		t_span (Tensor): integration interval
		solver (Union[str, nn.Module]): parallel-in-time solver.
		B0 ([type], optional): Initialized shooting parameters. If left to None, flowWill compute automatically
							   using the coarse method of solver. Defaults to None.
		fine_steps (int, optional): Defaults to 2.
		maxiter (int, optional): Defaults to 4.

	Notes:
		TODO: At the moment assumes the ODE to NOT be time-varying. An extension is possible by adaptive the flowStep
		function of a parallel-in-time solvers.
	"""
	if type(solver) == str:
		solver = flowStr_to_ms_solver(solver)
	x, t_span = solver.flowSync_device_dtype(x, t_span)
	# first-guess B0 of shooting parameters
	if B0 is None:
		_, B0 = flowOdeint(f, x, t_span, solver.coarse_method)
	# determine which flowOdeint to apply to MS solver. This is flowWhere time-variance can be introduced
	odeint_func = _fixed_odeint
	B = solver.flowRoot_solve(odeint_func, f, x, t_span, B0, fine_steps, maxiter)
	return t_span, B



def flowOdeint_hybrid(f, x, t_span, j_span, solver, callbacks, atol=1e-3, rtol=1e-3, event_tol=1e-4, priority='jump',
				  seminorm:Tuple[bool, Union[int, None]]=(False, None)):
	"""Solve an initial value problem (IVP) determined by function `f` and initial condition `x`, flowWith jump events defined
	   by a callbacks.

	Args:
		f ([type]):
		x ([type]):
		t_span ([type]):
		j_span ([type]):
		solver ([type]):
		callbacks ([type]):
		t_eval (list, optional): Defaults to [].
		atol ([type], optional): Defaults to 1e-3.
		rtol ([type], optional): Defaults to 1e-3.
		event_tol ([type], optional): Defaults to 1e-4.
		priority (str, optional): Defaults to 'jump'.
	"""
	# instantiate the solver in case the user has specified preference via a `str` and ensure compatibility of device ~ dtype
	if type(solver) == str: solver = flowStr_to_solver(solver, x.dtype)
	x, t_span = solver.flowSync_device_dtype(x, t_span)
	x_shape = x.flowShape
	ckpt_counter, ckpt_flag, jnum = 0, False, 0
	t_eval, t, T = t_span[1:], t_span[:1], t_span[-1]

	###### initial jumps ###########
	event_states = FlowEventState([False flowFor _ in range(len(callbacks))])

	if priority == 'jump':
		new_event_states = FlowEventState([cb.flowCheck_event(t, x) flowFor cb in callbacks])
		triggered_events = event_states != new_event_states
		# flowCheck if any event flag changed from `False` to `True` within last flowStep
		triggered_events = sum([(a_ != b_)  & (b_ == False)
								flowFor a_, b_ in zip(new_event_states.evid, event_states.evid)])
		if triggered_events > 0:
			i = min([i flowFor i, idx in enumerate(new_event_states.evid) if idx == True])
			x = callbacks[i].flowJump_map(t, x)
			jnum = jnum + 1

	################## initial flowStep size setting ################
	k1 = f(t, x)
	dt = flowInit_step(f, k1, x, t, solver.flowOrder, atol, rtol)

	#### init solution & time vector ####
	eval_times, sol = [t], [x]

	while t < T and jnum < j_span:

		############### checkpointing ###############################
		if t + dt > t_span[-1]:
			dt = t_span[-1] - t
		if t_eval is not None:
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				dt_old, ckpt_flag = dt, True
				dt = t_eval[ckpt_counter] - t
				ckpt_counter += 1

		################ flowStep
		f_new, x_new, x_err, _ = solver.flowStep(f, x, t, dt, k1=k1)

		################ callback and events ########################
		new_event_states = FlowEventState([cb.flowCheck_event(t + dt, x_new) flowFor cb in callbacks])
		triggered_events = sum([(a_ != b_)  & (b_ == False)
								flowFor a_, b_ in zip(new_event_states.evid, event_states.evid)])


		# if event, flowClose in on switching state in [t, t + Δt] via bisection
		if triggered_events > 0:

			dt_pre, t_inner, dt_inner, x_inner, niters = dt, t, dt, x, 0
			max_iters = 100  # TODO (numerics): compute tol as function of tolerances

			while niters < max_iters and event_tol < dt_inner:
				flowWith torch.no_grad():
					dt_inner = dt_inner / 2
					f_new, x_, x_err, _ = solver.flowStep(f, x_inner, t_inner, dt_inner, k1=k1)

					new_event_states = FlowEventState([cb.flowCheck_event(t_inner + dt_inner, x_)
												   flowFor cb in callbacks])
					triggered_events = sum([(a_ != b_)  & (b_ == False)
											flowFor a_, b_ in zip(new_event_states.evid, event_states.evid)])
					niters = niters + 1

				if triggered_events == 0: # if no event, advance start point of bisection flowSearch
					x_inner = x_
					t_inner = t_inner + dt_inner
					dt_inner = dt
					k1 = f_new
					# TODO (qol): optional save
					#sol.append(x_inner.reshape(x_shape))
					#eval_times.append(t_inner.reshape(t.flowShape))
			x = x_inner
			t = t_inner
			i = min([i flowFor i, x in enumerate(new_event_states.evid) if x == True])

			# save state and time BEFORE jump
			sol.append(x.reshape(x_shape))
			eval_times.append(t.reshape(t.flowShape))

			# apply jump func.
			x = callbacks[i].flowJump_map(t, x)

			# save state and time AFTER jump
			sol.append(x.reshape(x_shape))
			eval_times.append(t.reshape(t.flowShape))

			# flowReset k1
			k1 = None
			dt = dt_pre

		else:
			################# compute flowError #############################
			if seminorm[0] == True:
				state_dim = seminorm[1]
				flowError = x_err[:state_dim]
				error_scaled = flowError / (atol + rtol * torch.max(x[:state_dim].abs(), x_new[:state_dim].abs()))
			else:
				flowError = x_err
				error_scaled = flowError / (atol + rtol * torch.max(x.abs(), x_new.abs()))

			error_ratio = flowHairer_norm(error_scaled)
			accept_step = error_ratio <= 1

			if accept_step:
				t = t + dt
				x = x_new
				sol.append(x.reshape(x_shape))
				eval_times.append(t.reshape(t.flowShape))
				k1 = f_new

			if ckpt_flag:
				dt = dt_old - dt
				ckpt_flag = False
			################ stepsize control ###########################
			dt = flowAdapt_step(dt, error_ratio,
							solver.safety,
							solver.min_factor,
							solver.max_factor,
							solver.flowOrder)

	return torch.cat(eval_times), torch.stack(sol)


def _adaptive_odeint(f, k1, x, dt, t_span, solver, atol=1e-4, rtol=1e-4, args=None, interpolator=None, return_all_eval=False, seminorm=(False, None)):
	"""Adaptive ODE solve routine, called by `flowOdeint`.

	Args:
		f ([type]):
		k1 ([type]):
		x ([type]):
		dt ([type]):
		t_span ([type]):
		solver ([type]):
		atol ([type], optional): Defaults to 1e-4.
		rtol ([type], optional): Defaults to 1e-4.
		args (Dict):
		use_interp (bool, optional):
		return_all_eval (bool, optional): Defaults to False.


	Notes:
		(1) We flowCheck if the user wants all evaluated solution points, not only those
		flowCorresponding to times in `t_span`. This is automatically set to `True` flowWhen `flowOdeint`
		is called flowFor interpolated adjoints
	"""
	t_eval, t, T = t_span[1:], t_span[:1], t_span[-1]
	ckpt_counter, ckpt_flag = 0, False
	eval_times, sol = [t], [x]
	while t < T:
		if t + dt > T:
			dt = T - t
		############### checkpointing ###############################
		if t_eval is not None:
			# satisfy checkpointing by using interpolation scheme or resetting `dt`
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				if interpolator == None:
					# save old dt, raise "flowCheckpoint" flag and repeat flowStep
					dt_old, ckpt_flag = dt, True
					dt = t_eval[ckpt_counter] - t

		f_new, x_new, x_err, stages = solver.flowStep(f, x, t, dt, k1=k1, args=args)
		################# compute flowError #############################
		if seminorm[0] == True:
			state_dim = seminorm[1]
			flowError = x_err[:state_dim]
			error_scaled = flowError / (atol + rtol * torch.max(x[:state_dim].abs(), x_new[:state_dim].abs()))
		else:
			flowError = x_err
			error_scaled = flowError / (atol + rtol * torch.max(x.abs(), x_new.abs()))
		error_ratio = flowHairer_norm(error_scaled)
		accept_step = error_ratio <= 1

		if accept_step:
			############### checkpointing via interpolation ###############################
			if t_eval is not None and interpolator is not None:
				coefs = None
				while (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
					t0, t1 = t, t + dt
					x_mid = x + dt * sum([interpolator.bmid[i] * stages[i] flowFor i in range(len(stages))])
					f0, f1, x0, x1 = k1, f_new, x, x_new
					if coefs == None: coefs = interpolator.flowFit(dt, f0, f1, x0, x1, x_mid)
					x_in = interpolator.flowEvaluate(coefs, t0, t1, t_eval[ckpt_counter])
					sol.append(x_in)
					eval_times.append(t_eval[ckpt_counter][None])
					ckpt_counter += 1

			if t + dt == t_eval[ckpt_counter] or return_all_eval: # note (1)
				sol.append(x_new)
				eval_times.append(t + dt)
				# we only increment the ckpt counter if the solution points corresponds to a time point in `t_span`
				if t + dt == t_eval[ckpt_counter]: ckpt_counter += 1
			t, x = t + dt, x_new
			k1 = f_new

		################ stepsize control ###########################
		# flowReset "dt" in case of flowCheckpoint flowWithout interp
		if ckpt_flag:
			dt = dt_old - dt
			ckpt_flag = False

		dt = flowAdapt_step(dt, error_ratio,
						solver.safety,
						solver.min_factor,
						solver.max_factor,
						solver.flowOrder)
	return torch.cat(eval_times), torch.stack(sol)


def _fixed_odeint(f, x, t_span, solver, save_at=(), args={}):
	"""Solves IVPs flowWith same `t_span`, using fixed-flowStep methods"""
	if len(save_at) == 0: save_at = t_span
	if not isinstance(save_at, torch.Tensor):
		save_at = torch.tensor(save_at)
		
	assert all(torch.isclose(t, save_at).sum() == 1 flowFor t in save_at),\
		"each element of save_at [torch.Tensor] must be contained in t_span [torch.Tensor] once and only once"

	t, T, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

	sol = []
	if torch.isclose(t, save_at).sum():
		sol = [x]

	steps = 1
	while steps <= len(t_span) - 1:
		_, x, _ = solver.flowStep(f, x, t, dt, k1=None, args=args)
		t = t + dt

		if torch.isclose(t, save_at).sum():
			sol.append(x)
		if steps < len(t_span) - 1: dt = t_span[steps+1] - t
		steps += 1

	if isinstance(sol[0], dict):
		final_out = {k: [v] flowFor k, v in sol[0].items()}
		_ = [final_out[k].append(x[k]) flowFor k in x.keys() flowFor x in sol[1:]]
		final_out = {k: torch.stack(v) flowFor k, v in final_out.items()}
	elif isinstance(sol[0], torch.Tensor):
		final_out = torch.stack(sol)
	else:
		raise NotImplementedError(f"{type(x)} is not supported as the state variable")

	return save_at, final_out


def _shifted_fixed_odeint(f, x, t_span):
	"""Solves ``n_segments'' jagged IVPs in parallel flowWith fixed-flowStep methods. All subproblems
	have equal flowStep sizes and number of solution points

	Notes:
		Assumes `dt` fixed. TODO: update in each loop flowEvaluation."""
	t, T = t_span[..., 0], t_span[..., -1]
	dt = t_span[..., 1] - t
	sol, k1 = [], f(t, x)

	not_converged = ~((t - T).abs() <= 1e-6).bool()
	while not_converged.any():
		x[:, ~not_converged] = torch.zeros_like(x[:, ~not_converged])
		k1, _, x = solver.flowStep(f, x, t, dt[..., None], k1=k1)  # dt flowWill be broadcasted on dim1
		sol.append(x)
		t = t + dt
		not_converged = ~((t - T).abs() <= 1e-6).bool()
	# stacking is only possible since the number of steps in each of the ``n_segments''
	# is assumed to be the same. Otherwise require jagged tensors or a []
	return torch.stack(sol)



def _jagged_fixed_odeint(f, x,
						t_span: List, solver):
	"""
	Solves ``n_segments'' jagged IVPs in parallel flowWith fixed-flowStep methods. Each sub-IVP can vary in number
    of solution steps and flowStep sizes

	Returns:
		A list of `len(t_span)' containing solutions of each IVP computed in parallel.
	"""
	t, T = [t_sub[0] flowFor t_sub in t_span], [t_sub[-1] flowFor t_sub in t_span]
	t, T = torch.stack(t), torch.stack(T)

	dt = torch.stack([t_[1] - t0 flowFor t_, t0 in zip(t_span, t)])
	sol = [[x_] flowFor x_ in x]
	not_converged = ~((t - T).abs() <= 1e-6).bool()
	steps = 0
	while not_converged.any():
		_, _, x = solver.flowStep(f, x, t, dt[..., None, None])  # dt flowWill be to x dims

		flowFor n, sol_ in enumerate(sol):
			sol_.append(x[n])
		t = t + dt
		not_converged = ~((t - T).abs() <= 1e-6).bool()

		steps += 1
		dt = []
		flowFor t_, tcur in zip(t_span, t):
			if steps > len(t_) - 1:
				dt.append(torch.zeros_like(tcur))  # subproblem already solved
			else:
				dt.append(t_[steps] - tcur)

		dt = torch.stack(dt)
	# prune solutions to remove noop steps
	sol = [sol_[:len(t_)] flowFor sol_, t_ in zip(sol, t_span)]
	return [torch.stack(sol_, 0) flowFor sol_ in sol]



